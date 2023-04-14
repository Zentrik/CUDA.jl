## gpucompiler interface implementation

struct CUDACompilerParams <: AbstractCompilerParams end
const CUDACompilerConfig = CompilerConfig{PTXCompilerTarget, CUDACompilerParams}
const CUDACompilerJob = CompilerJob{PTXCompilerTarget,CUDACompilerParams}

GPUCompiler.runtime_module(@nospecialize(job::CUDACompilerJob)) = CUDA

# filter out functions from libdevice and cudadevrt
GPUCompiler.isintrinsic(@nospecialize(job::CUDACompilerJob), fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(fn)},
           job, fn) ||
    fn == "__nvvm_reflect" || startswith(fn, "cuda")

function GPUCompiler.link_libraries!(@nospecialize(job::CUDACompilerJob), mod::LLVM.Module,
                                     undefined_fns::Vector{String})
    invoke(GPUCompiler.link_libraries!,
           Tuple{CompilerJob{PTXCompilerTarget}, typeof(mod), typeof(undefined_fns)},
           job, mod, undefined_fns)
    link_libdevice!(mod, job.config.target.cap, undefined_fns)
end

GPUCompiler.method_table(@nospecialize(job::CUDACompilerJob)) = method_table

GPUCompiler.kernel_state_type(job::CUDACompilerJob) = KernelState


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per context
const _compiler_caches = Dict{CuContext, Dict{UInt, Any}}();
function compiler_cache(ctx::CuContext)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{UInt, Any}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, CUDACompilerConfig}()
function compiler_config(dev; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    # determine the toolchain (cached, because this is slow)
    if !isassigned(_toolchain)
        _toolchain[] = supported_toolchain()
    end
    toolchain = _toolchain[]::@NamedTuple{cap::Vector{VersionNumber}, ptx::Vector{VersionNumber}}

    # select the highest capability that is supported by both the toolchain and device
    caps = filter(toolchain_cap -> toolchain_cap <= capability(dev), toolchain.cap)
    isempty(caps) &&
        error("Your $(name(dev)) GPU with capability v$(capability(dev)) is not supported by the available toolchain")
    cap = maximum(caps)

    # select the PTX ISA we assume to be available
    # (we actually only need 6.2, but NVPTX doesn't support that)
    ptx = v"6.3"

    # we need to take care emitting LLVM instructions like `unreachable`, which
    # may result in thread-divergent control flow that older `ptxas` doesn't like.
    # see e.g. JuliaGPU/CUDAnative.jl#4
    unreachable = true
    if cap < v"7" || runtime_version() < v"11.3"
        unreachable = false
    end

    # there have been issues with emitting PTX `exit` instead of `trap` as well,
    # see e.g. JuliaGPU/CUDA.jl#431 and NVIDIA bug #3231266 (but since switching
    # to the toolkit's `ptxas` that specific machine/GPU now _requires_ exit...)
    exitable = true
    if cap < v"7"
        exitable = false
    end

    # NVIDIA bug #3600554: ptxas segfaults with our debug info, fixed in 11.7
    debuginfo = runtime_version() >= v"11.7"

    # create GPUCompiler objects
    target = PTXCompilerTarget(; cap, ptx, debuginfo, unreachable, exitable, kwargs...)
    params = CUDACompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    JuliaContext() do ctx
        compile(job, ctx)
    end
end
function compile(@nospecialize(job::CompilerJob), ctx)
    # lower to PTX
    asm, meta = GPUCompiler.compile(:asm, job; ctx)

    # remove extraneous debug info on lower debug levels
    if Base.JLOptions().debug_level < 2
        # LLVM sets `.target debug` as soon as the debug emission kind isn't NoDebug. this
        # is unwanted, as the flag makes `ptxas` behave as if `--device-debug` were set.
        # ideally, we'd need something like LocTrackingOnly/EmitDebugInfo from D4234, but
        # that got removed in favor of NoDebug in D18808, seemingly breaking the use case of
        # only emitting `.loc` instructions...
        #
        # according to NVIDIA, "it is fine for PTX producers to produce debug info but not
        # set `.target debug` and if `--device-debug` isn't passed, PTXAS will compile in
        # release mode".
        asm = replace(asm, r"(\.target .+), debug" => s"\1")
    end

    # check if we'll need the device runtime
    undefined_fs = filter(collect(functions(meta.ir))) do f
        isdeclaration(f) && !LLVM.isintrinsic(f)
    end
    intrinsic_fns = ["vprintf", "malloc", "free", "__assertfail",
                     "__nvvm_reflect" #= TODO: should have been optimized away =#]
    needs_cudadevrt = !isempty(setdiff(LLVM.name.(undefined_fs), intrinsic_fns))

    # find externally-initialized global variables; we'll access those using CUDA APIs.
    external_gvars = filter(isextinit, collect(globals(meta.ir))) .|> LLVM.name

    # prepare invocations of CUDA compiler tools
    ptxas_opts = String[]
    nvlink_opts = String[]
    ## debug flags
    if Base.JLOptions().debug_level == 1
        push!(ptxas_opts, "--generate-line-info")
    elseif Base.JLOptions().debug_level >= 2
        push!(ptxas_opts, "--device-debug")
        push!(nvlink_opts, "--debug")
    end
    ## relocatable device code
    if needs_cudadevrt
        push!(ptxas_opts, "--compile-only")
    end

    arch = "sm_$(job.config.target.cap.major)$(job.config.target.cap.minor)"

    # compile to machine code
    # NOTE: we use tempname since mktemp doesn't support suffixes, and mktempdir is slow
    ptx_input = tempname(cleanup=false) * ".ptx"
    ptxas_output = tempname(cleanup=false) * ".cubin"
    write(ptx_input, asm)

    # we could use the driver's embedded JIT compiler, but that has several disadvantages:
    # 1. fixes and improvements are slower to arrive, by using `ptxas` we only need to
    #    upgrade the toolkit to get a newer compiler;
    # 2. version checking is simpler, we otherwise need to use NVML to query the driver
    #    version, which is hard to correlate to PTX JIT improvements;
    # 3. if we want to be able to use newer (minor upgrades) of the CUDA toolkit on an
    #    older driver, we should use the newer compiler to ensure compatibility.
    append!(ptxas_opts, [
        "--verbose",
        "--gpu-name", arch,
        "--output-file", ptxas_output,
        ptx_input
    ])
    proc, log = run_and_collect(`$(ptxas()) $ptxas_opts`)
    log = strip(log)
    if !success(proc)
        reason = proc.termsignal > 0 ? "ptxas received signal $(proc.termsignal)" :
                                       "ptxas exited with code $(proc.exitcode)"
        msg = "Failed to compile PTX code ($reason)"
        msg *= "\nInvocation arguments: $(join(ptxas_opts, ' '))"
        if !isempty(log)
            msg *= "\n" * log
        end
        msg *= "\nIf you think this is a bug, please file an issue and attach $(ptx_input)"
        error(msg)
    elseif !isempty(log)
        @debug "PTX compiler log:\n" * log
    end
    rm(ptx_input)

    # link device libraries, if necessary
    #
    # this requires relocatable device code, which prevents certain optimizations and
    # hurts performance. as such, we only do so when absolutely necessary.
    # TODO: try LTO, `--link-time-opt --nvvmpath /opt/cuda/nvvm`.
    #       fails with `Ignoring -lto option because no LTO objects found`
    if needs_cudadevrt
        nvlink_output = tempname(cleanup=false) * ".cubin"
        append!(nvlink_opts, [
            "--verbose", "--extra-warnings",
            "--arch", arch,
            "--library-path", dirname(libcudadevrt),
            "--library", "cudadevrt",
            "--output-file", nvlink_output,
            ptxas_output
        ])
        proc, log = run_and_collect(`$(nvlink()) $nvlink_opts`)
        log = strip(log)
        if !success(proc)
            reason = proc.termsignal > 0 ? "nvlink received signal $(proc.termsignal)" :
                                           "nvlink exited with code $(proc.exitcode)"
            msg = "Failed to link PTX code ($reason)"
            msg *= "\nInvocation arguments: $(join(nvlink_opts, ' '))"
            if !isempty(log)
                msg *= "\n" * log
            end
            msg *= "\nIf you think this is a bug, please file an issue and attach $(ptxas_output)"
            error(msg)
        elseif !isempty(log)
            @debug "PTX linker info log:\n" * log
        end
        rm(ptxas_output)

        image = read(nvlink_output)
        rm(nvlink_output)
    else
        image = read(ptxas_output)
        rm(ptxas_output)
    end

    return (image, entry=LLVM.name(meta.entry), external_gvars)
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    # load as an executable kernel object
    ctx = context()
    mod = CuModule(compiled.image)
    CuFunction(mod, compiled.entry)
end


## helpers

# run a binary and collect all relevant output
function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)

    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))

    return proc, log
end



## opaque closures

# TODO: once stabilised, move bits of this into GPUCompiler.jl

using Core.Compiler: IRCode
using Core: CodeInfo, MethodInstance, CodeInstance, LineNumberNode

struct OpaqueClosure{F, E, A, R}    # func, env, args, ret
    env::E
end

# XXX: because we can't call functions from other CUDA modules, we effectively need to
#      recompile when the target function changes. this, and because of how GPUCompiler's
#      deferred compilation mechanism currently works, is why we have `F` as a type param.

# XXX: because of GPU code requiring specialized signatures, we also need to recompile
#      when the environment or argument types change. together with the above, this
#      negates much of the benefit of opaque closures.

# TODO: support for constructing an opaque closure from source code

# TODO: complete support for passing an environment. this probably requires a split into
#       host and device structures to, e.g., root a CuArray and pass a CuDeviceArray.

function compute_ir_rettype(ir::IRCode)
    rt = Union{}
    for i = 1:length(ir.stmts)
        stmt = ir.stmts[i][:inst]
        if isa(stmt, Core.Compiler.ReturnNode) && isdefined(stmt, :val)
            rt = Core.Compiler.tmerge(Core.Compiler.argextype(stmt.val, ir), rt)
        end
    end
    return Core.Compiler.widenconst(rt)
end

function compute_oc_signature(ir::IRCode, nargs::Int, isva::Bool)
    argtypes = Vector{Any}(undef, nargs)
    for i = 1:nargs
        argtypes[i] = Core.Compiler.widenconst(ir.argtypes[i+1])
    end
    if isva
        lastarg = pop!(argtypes)
        if lastarg <: Tuple
            append!(argtypes, lastarg.parameters)
        else
            push!(argtypes, Vararg{Any})
        end
    end
    return Tuple{argtypes...}
end

function OpaqueClosure(ir::IRCode, @nospecialize env...; isva::Bool = false)
    # NOTE: we need ir.argtypes[1] == typeof(env)
    ir = Core.Compiler.copy(ir)
    nargs = length(ir.argtypes)-1
    sig = compute_oc_signature(ir, nargs, isva)
    rt = compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    src.slotnames = Base.fill(:none, nargs+1)
    src.slotflags = Base.fill(zero(UInt8), length(ir.argtypes))
    src.slottypes = copy(ir.argtypes)
    src.rettype = rt
    src = Core.Compiler.ir_to_codeinf!(src, ir)
    config = compiler_config(device(); kernel=false)
    return generate_opaque_closure(config, src, sig, rt, nargs, isva, env...)
end

function OpaqueClosure(src::CodeInfo, @nospecialize env...)
    src.inferred || throw(ArgumentError("Expected inferred src::CodeInfo"))
    mi = src.parent::Core.MethodInstance
    sig = Base.tuple_type_tail(mi.specTypes)
    method = mi.def::Method
    nargs = method.nargs-1
    isva = method.isva
    config = compiler_config(device(); kernel=false)
    return generate_opaque_closure(config, src, sig, src.rettype, nargs, isva, env...)
end

function generate_opaque_closure(config::CompilerConfig, src::CodeInfo,
                                 @nospecialize(sig), @nospecialize(rt),
                                 nargs::Int, isva::Bool, @nospecialize env...;
                                 mod::Module=@__MODULE__,
                                 file::Union{Nothing,Symbol}=nothing, line::Int=0)
    # create a method (like `jl_make_opaque_closure_method`)
    meth = ccall(:jl_new_method_uninit, Ref{Method}, (Any,), Main)
    meth.sig = Tuple
    meth.isva = isva                # XXX: probably not supported?
    meth.is_for_opaque_closure = 0  # XXX: do we want this?
    meth.name = Symbol("opaque gpu closure")
    meth.nargs = nargs + 1
    meth.file = something(file, Symbol())
    meth.line = line
    ccall(:jl_method_set_source, Nothing, (Any, Any), meth, src)

    # look up a method instance and create a compiler job
    full_sig = Tuple{typeof(env), sig.parameters...}
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), meth, full_sig, Core.svec())
    job = CompilerJob(mi, config)   # this captures the current world age

    # create a code instance and store it in the cache
    ci = CodeInstance(mi, rt, C_NULL, src, Int32(0), meth.primary_world, typemax(UInt),
                      UInt32(0), UInt32(0), nothing, UInt8(0))
    Core.Compiler.setindex!(GPUCompiler.ci_cache(job), ci, mi)

    id = length(GPUCompiler.deferred_codegen_jobs) + 1
    GPUCompiler.deferred_codegen_jobs[id] = job
    return OpaqueClosure{id, typeof(env), sig, rt}(env)
end

# device-side call to an opaque closure
function (oc::OpaqueClosure{F})(a, b) where F
    ptr = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), F)
    assume(ptr != C_NULL)
    return ccall(ptr, Int, (Int, Int), a, b)
end
