# discovering binary CUDA dependencies

using CompilerSupportLibraries_jll
using LazyArtifacts
import Libdl

const dependency_lock = ReentrantLock()

# lazily initialize a Ref containing a path to a library.
# the arguments to this macro is the name of the ref, an expression to populate it
# (possibly returning `nothing` if the library wasn't found), and an optional initialization
# hook to be executed after successfully discovering the library and setting the ref.
macro initialize_ref(ref, ex, hook=:())
    quote
        ref = $ref

        # test and test-and-set
        if !isassigned(ref)
            Base.@lock dependency_lock begin
                if !isassigned(ref)
                    val = $ex
                    if val === nothing && !(eltype($ref) <: Union{Nothing,<:Any})
                        error($"Could not find a required library")
                    end
                    $ref[] = $ex
                    if val !== nothing
                        $hook
                    end
                end
            end
        end

        $ref[]
    end
end


#
# CUDA toolkit
#

export toolkit

abstract type AbstractToolkit end

struct ArtifactToolkit <: AbstractToolkit
    version::VersionNumber
    artifact::String
end

struct LocalToolkit <: AbstractToolkit
    version::VersionNumber
    dirs::Vector{String}
end

const __toolkit = Ref{AbstractToolkit}()

function toolkit()
    @initialize_ref __toolkit begin
        toolkit = nothing

        # CI runs in a well-defined environment, so prefer a local CUDA installation there
        if getenv("CI", false) && !haskey(ENV, "JULIA_CUDA_USE_BINARYBUILDER")
            toolkit = find_local_cuda()
        end

        if toolkit === nothing && getenv("JULIA_CUDA_USE_BINARYBUILDER", true)
            toolkit = find_artifact_cuda()
        end

        # if the user didn't specifically request an artifact version, look for a local installation
        if toolkit === nothing && !haskey(ENV, "JULIA_CUDA_VERSION")
            toolkit = find_local_cuda()
        end

        if toolkit === nothing
            error("Could not find a suitable CUDA installation")
        end

        toolkit
    end CUDA.__init_toolkit__()
    __toolkit[]::Union{ArtifactToolkit,LocalToolkit}
end

# workaround @artifact_str eagerness on unsupported platforms by passing a variable
function cuda_artifact(id, cuda::VersionNumber)
    platform = Base.BinaryPlatforms.HostPlatform()
    platform.tags["cuda"] = "$(cuda.major).$(cuda.minor)"
    @artifact_str(id, platform)
end

# NOTE: we don't use autogenerated JLLs, because we have multiple artifacts and need to
#       decide at run time (i.e. not via package dependencies) which one to use.
const cuda_toolkits = [
    (release=v"11.4", version=v"11.4.0",   preferred=false),
    (release=v"11.3", version=v"11.3.1",   preferred=true),
    (release=v"11.2", version=v"11.2.2",   preferred=true),
    (release=v"11.1", version=v"11.1.1",   preferred=true),
    (release=v"11.0", version=v"11.0.3",   preferred=true),
    (release=v"10.2", version=v"10.2.89",  preferred=true),
    (release=v"10.1", version=v"10.1.243", preferred=true),
]

function find_artifact_cuda()
    @debug "Trying to use artifacts..."

    # select compatible artifacts
    if haskey(ENV, "JULIA_CUDA_VERSION")
        wanted = VersionNumber(ENV["JULIA_CUDA_VERSION"])
        @debug "Selecting artifacts based on requested $wanted"
        candidate_toolkits = filter(cuda_toolkits) do toolkit
            toolkit.release == wanted || toolkit.version == wanted
        end
        isempty(candidate_toolkits) && @debug "Requested CUDA version $wanted is not provided by any artifact"
    else
        driver_release = CUDA.release()
        @debug "Selecting artifacts based on driver compatibility $driver_release"
        candidate_toolkits = filter(cuda_toolkits) do toolkit
            toolkit.preferred &&
                (toolkit.release <= driver_release ||
                 # CUDA 11: Enhanced Compatibility (aka. semver)
                 (driver_release >= v"11" &&
                  toolkit.release.major <= driver_release.major))
        end
        isempty(candidate_toolkits) && @debug "CUDA driver compatibility $driver_release is not compatible with any artifact"
    end

    # download and install
    artifact = nothing
    for cuda in sort(candidate_toolkits; rev=true, by=toolkit->toolkit.version)
        try
            artifact = (version=cuda.version, dir=cuda_artifact("CUDA", cuda.version))
            break
        catch ex
            @debug "Could not load the CUDA $(cuda.version) artifact" exception=(ex,catch_backtrace())
        end
    end
    if artifact == nothing
        @debug "Could not find a compatible artifact."
        return nothing
    end

    @debug "Using CUDA $(artifact.version) from an artifact at $(artifact.dir)"
    return ArtifactToolkit(artifact.version, artifact.dir)
end

function find_local_cuda()
    @debug "Trying to use local installation..."

    dirs = find_toolkit()
    let path = find_cuda_binary("nvdisasm", dirs)
        if path === nothing
            @debug "Could not find nvdisasm"
            return nothing
        end
        __nvdisasm[] = path
    end

    version = parse_toolkit_version("nvdisasm", __nvdisasm[])
    if version === nothing
        return nothing
    end

    # CUDA 11.1 Update 1 ships the same `nvdisasm` as 11.1 GA, so look at the version of
    # CUSOLVER (which has a handle-less version getter that does not initialize)
    # to be sure which CUDA we're dealing with (it only matters for CUPTI).
    if version == v"11.1.0"
        temp_libcusolver = find_cuda_library("cusolver", dirs, v"11.1.0")
        if temp_libcusolver === nothing
            temp_libcusolver = find_cuda_library("cusolver", dirs, v"11.1.1")
        end
        if temp_libcusolver === nothing
            @debug "Could not disambiguate CUDA 11.1 from Update 1 due to not finding CUSOLVER"
        else
            # nothing is initialized at this point, so we need to use raw ccalls.
            Libdl.dlopen(temp_libcusolver) do lib
                fun = Libdl.dlsym(lib, :cusolverGetVersion)
                @assert fun != C_NULL

                cusolver_version = Ref{Cint}()
                @assert 0 == ccall(fun, Cint, (Ref{Cint},), cusolver_version)
                if cusolver_version[] == 11001
                    version = v"11.1.1"
                elseif cusolver_version[] != 11000
                    @debug "Could not disambiguate CUDA 11.1 from Update 1 with CUSOLVER version $(cusolver_version[])"
                end
            end

            __libcusolver[] = temp_libcusolver
        end
    end

    # same with CUDA 11.3 / 11.3.1, but at least ptxas differs there
    if version == v"11.3.0"
        let path = find_cuda_binary("ptxas", dirs)
            if path === nothing
                @debug "Could not find ptxas"
                return nothing
            end
            __ptxas[] = path
        end
        version = parse_toolkit_version("ptxas", __ptxas[])
    end

    @debug "Found local CUDA $(version) at $(join(dirs, ", "))"
    return LocalToolkit(version, dirs)
end


## properties

export toolkit_origin, toolkit_version, toolkit_release

"""
    toolkit_origin()

Returns the origin of the CUDA toolkit in use (either :artifact, or :local).
"""
toolkit_origin() = toolkit_origin(toolkit())::Symbol
toolkit_origin(::ArtifactToolkit) = :artifact
toolkit_origin(::LocalToolkit) = :local

"""
    toolkit_version()

Returns the version of the CUDA toolkit in use.
"""
toolkit_version() = toolkit().version::VersionNumber

"""
    toolkit_release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
toolkit_release() = VersionNumber(toolkit_version().major, toolkit_version().minor)::VersionNumber


## binaries

export ptxas, nvlink, nvdisasm, compute_sanitizer, has_compute_sanitizer

# pxtas: used for compiling PTX to SASS
const __ptxas = Ref{String}()
function ptxas()
    @initialize_ref __ptxas begin
        find_binary(toolkit(), "ptxas")
    end
end

# nvlink: used for linking additional libraries
const __nvlink = Ref{String}()
function nvlink()
    @initialize_ref __nvlink begin
        find_binary(toolkit(), "nvlink")
    end
end

# nvdisasm: used for reflection (decompiling SASS code)
const __nvdisasm = Ref{String}()
function nvdisasm()
    @initialize_ref __nvdisasm begin
        find_binary(toolkit(), "nvdisasm")
    end
end

# compute-santizer: used by the test suite
const __compute_sanitizer = Ref{Union{Nothing,String}}()
function compute_sanitizer(throw_error::Bool=true)
    @initialize_ref __compute_sanitizer begin
        find_binary(toolkit(), "compute-sanitizer"; optional=true)
    end
end
has_compute_sanitizer() = compute_sanitizer(throw_error=false) !== nothing

artifact_binary(artifact_dir, name) = joinpath(artifact_dir, "bin", Sys.iswindows() ? "$name.exe" : name)

function find_binary(cuda::ArtifactToolkit, name; optional=false)
    path = artifact_binary(cuda.artifact, name)
    if isfile(path)
        return path
    else
        optional ||
            error("""Could not find binary '$name' in $(dirname(path))!
                     This is a bug; please file an issue with a verbose directory listing of $(dirname(path)).""")
        return nothing
    end
end

function find_binary(cuda::LocalToolkit, name; optional=false)
    path = find_cuda_binary(name, cuda.dirs)
    if path !== nothing
        return path
    else
        optional || error("Could not find binary '$name' in your local CUDA installation.")
        return nothing
    end
end


## libraries

export libcublas, libcusparse, libcufft, libcurand, libcusolver,
       libcusolvermg, has_cusolvermg, libcupti, has_cupti, libnvtx, has_nvtx

const __libcublas = Ref{String}()
function libcublas()
    @initialize_ref __libcublas begin
        cuda = toolkit()

        # HACK: eagerly load cublasLt, required by cublas (but with the same version), to
        #       prevent a local CUDA from messing with our artifacts (JuliaGPU/CUDA.jl#609)
        if cuda isa ArtifactToolkit && cuda.version >= v"10.1"
            find_library(cuda, "cublasLt")
        end

        find_library(cuda, "cublas")
    end CUDA.CUBLAS.__runtime_init__()
end

const __libcusparse = Ref{String}()
function libcusparse()
    @initialize_ref __libcusparse begin
        find_library(toolkit(), "cusparse")
    end
end

const __libcufft = Ref{String}()
function libcufft()
    @initialize_ref __libcufft begin
        find_library(toolkit(), "cufft")
    end
end

const __libcurand = Ref{String}()
function libcurand()
    @initialize_ref __libcurand begin
        find_library(toolkit(), "curand")
    end
end

const __libcusolver = Ref{String}()
function libcusolver()
    @initialize_ref __libcusolver begin
        find_library(toolkit(), "cusolver")
    end
end

const __libcusolverMg = Ref{Union{String,Nothing}}()
function libcusolvermg(; throw_error::Bool=true)
     path = @initialize_ref __libcusolverMg begin
        if toolkit_version() < v"10.1"
            nothing
        else
            find_library(toolkit(), "cusolverMg")
        end
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as cuSolverMg is missing.")
    end
    path
end
has_cusolvermg() = libcusolvermg(throw_error=false) !== nothing

const __libcupti = Ref{Union{String,Nothing}}()
function libcupti(; throw_error::Bool=true)
    path = @initialize_ref __libcupti begin
        find_library(toolkit(), "cupti")
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUPTI is missing.")
    end
    path
end
has_cupti() = libcupti(throw_error=false) !== nothing

const __libnvtx = Ref{Union{String,Nothing}}()
function libnvtx(; throw_error::Bool=true)
    path = @initialize_ref __libnvtx begin
        find_library(toolkit(), "nvtx")
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as NVTX is missing.")
    end
    path
end
has_nvtx() = libnvtx(throw_error=false) !== nothing

function artifact_library(artifact, name, version)
    dir = joinpath(artifact, Sys.iswindows() ? "bin" : "lib")
    all_names = library_versioned_names(name, version)
    for name in all_names
        path = joinpath(dir, name)
        ispath(path) && return path
    end
    error("Could not find $name ($(join(all_names, ", ", " or "))) in $dir")
end

function artifact_cuda_library(artifact, library, toolkit_version)
    version = cuda_library_version(library, toolkit_version)
    name = get(cuda_library_names, library, library)
    artifact_library(artifact, name, version)
end

function find_library(cuda::ArtifactToolkit, name; optional=false)
    path = artifact_cuda_library(cuda.artifact, name, cuda.version)
    if isfile(path)
        Libdl.dlopen(path)
        return path
    else
        optional ||
            error("""Could not find library '$name' in $(dirname(path))!
                     This is a bug; please file an issue with a verbose directory listing of $(dirname(path)).""")
        return nothing
    end
end

function find_library(cuda::LocalToolkit, name; optional=false)
    path = find_cuda_library(name, cuda.dirs, cuda.version)
    if path !== nothing
        return path
    else
        optional || error("Could not find library '$name' in your local CUDA installation.")
        return nothing
    end
end


## other

export libdevice, libcudadevrt

const __libdevice = Ref{String}()
function libdevice()
    @initialize_ref __libdevice begin
        find_libdevice(toolkit())
    end
end

artifact_file(artifact_dir, path) = joinpath(artifact_dir, path)

function find_libdevice(cuda::ArtifactToolkit)
    path = artifact_file(cuda.artifact, joinpath("share", "libdevice", "libdevice.10.bc"))
    if isfile(path)
        return path
    else
        error("""Could not find libdevice in $(dirname(path))!
                 This is a bug; please file an issue with a verbose directory listing of $(dirname(path)).""")
    end
end

function find_libdevice(cuda::LocalToolkit)
    path = find_libdevice(cuda.dirs)
    if path !== nothing
        return path
    else
        error("Could not find libdevice in your local CUDA installation.")
    end
end

const __libcudadevrt = Ref{String}()
function libcudadevrt()
    @initialize_ref __libcudadevrt begin
        find_libcudadevrt(toolkit())
    end
end

artifact_static_library(artifact_dir, name) = joinpath(artifact_dir, "lib", Sys.iswindows() ? "$name.lib" : "lib$name.a")

function find_libcudadevrt(cuda::ArtifactToolkit)
    path = artifact_static_library(cuda.artifact, "cudadevrt")
    if isfile(path)
        return path
    else
        error("""Could not find libcudadevrt in $(dirname(path))!
                 This is a bug; please file an issue with a verbose directory listing of $(dirname(path)).""")
    end
end

function find_libcudadevrt(cuda::LocalToolkit)
    path = find_libcudadevrt(cuda.dirs)
    if path !== nothing
        return path
    else
        error("Could not find libcudadevrt in your local CUDA installation.")
    end
end


#
# CUDNN
#

export libcudnn, has_cudnn

const __libcudnn = Ref{Union{String,Nothing}}()
function libcudnn(; throw_error::Bool=true)
    path = @initialize_ref __libcudnn begin
        find_cudnn(toolkit(), v"8")
    end CUDA.CUDNN.__runtime_init__()
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUDNN is missing.")
    end
    path
end
has_cudnn() = libcudnn(throw_error=false) !== nothing

function find_cudnn(cuda::ArtifactToolkit, version)
    artifact_dir = cuda_artifact("CUDNN", cuda.version)
    path = artifact_library(artifact_dir, "cudnn", version)
    if !isfile(path)
        error("""Could not find CUDNN in $(dirname(path))!
                 This is a bug; please file an issue with a verbose directory listing of $(dirname(path)).""")
    end

    # HACK: eagerly open CUDNN sublibraries to avoid dlopen discoverability issues
    for sublibrary in ("ops_infer", "ops_train",
                       "cnn_infer", "cnn_train",
                       "adv_infer", "adv_train")
        sublibrary_path = artifact_library(artifact_dir, "cudnn_$(sublibrary)", version)
        Libdl.dlopen(sublibrary_path)
    end

    @debug "Using CUDNN from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_cudnn(cuda::LocalToolkit, version)
    path = find_library("cudnn", version; locations=cuda.dirs)
    if path === nothing
        return nothing
    end

    # HACK: eagerly open CUDNN sublibraries to avoid dlopen discoverability issues
    for sublibrary in ("ops_infer", "ops_train",
                       "cnn_infer", "cnn_train",
                       "adv_infer", "adv_train")
        sublibrary_path = find_library("cudnn_$(sublibrary)", version; locations=cuda.dirs)
        sublibrary_path === nothing && error("Could not find local CUDNN sublibrary $sublibrary")
        Libdl.dlopen(sublibrary_path)
    end

    @debug "Using local CUDNN at $(path)"
    Libdl.dlopen(path)
    return path
end


#
# CUTENSOR
#

export libcutensor, has_cutensor

const __libcutensor = Ref{Union{String,Nothing}}()
function libcutensor(; throw_error::Bool=true)
    path = @initialize_ref __libcutensor begin
        version = Sys.iswindows() ? nothing : v"1"  # cutensor.dll is unversioned on Windows
        find_cutensor(toolkit(), version)
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUTENSOR is missing.")
    end
    path
end
has_cutensor() = libcutensor(throw_error=false) !== nothing

function find_cutensor(cuda::ArtifactToolkit, version)
    artifact_dir = cuda_artifact("CUTENSOR", cuda.version)
    path = artifact_library(artifact_dir, "cutensor", version)

    @debug "Using CUTENSOR from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_cutensor(cuda::LocalToolkit, version)
    path = find_library("cutensor", version; locations=cuda.dirs)
    if path === nothing
        path = find_library("cutensor"; locations=cuda.dirs)
    end
    if path === nothing
        return nothing
    end

    @debug "Using local CUTENSOR at $(path)"
    Libdl.dlopen(path)
    return path
end
