# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDA.MAJOR_VERSION),
                          cufftGetProperty(CUDA.MINOR_VERSION),
                          cufftGetProperty(CUDA.PATCH_LEVEL))

function cufftMakePlan(xtype::cufftType_t, xdims::Dims, region)
    nrank = length(region)
    sz = [xdims[i] for i in region]
    csz = copy(sz)
    csz[1] = div(sz[1],2) + 1
    batch = prod(xdims) ÷ prod(sz)

    # initialize the plan handle
    handle_ref = Ref{cufftHandle}()
    cufftCreate(handle_ref)
    handle = handle_ref[]

    # make the plan
    worksize_ref = Ref{Csize_t}()
    if (nrank == 1) && (batch == 1)
        cufftMakePlan1d(handle, sz[1], xtype, 1, worksize_ref)
    elseif (nrank == 2) && (batch == 1)
        cufftMakePlan2d(handle, sz[2], sz[1], xtype, worksize_ref)
    elseif (nrank == 3) && (batch == 1)
        cufftMakePlan3d(handle, sz[3], sz[2], sz[1], xtype, worksize_ref)
    else
        rsz = (length(sz) > 1) ? rsz = reverse(sz) : sz
        if ((region...,) == ((1:nrank)...,))
            # handle simple case ... simply! (for robustness)
           cufftMakePlanMany(handle, nrank, Cint[rsz...], C_NULL, 1, 1, C_NULL, 1, 1,
                             xtype, batch, worksize_ref)
        else
            if nrank==1 || all(diff(collect(region)) .== 1)
                # _stride: successive elements in innermost dimension
                # _dist: distance between first elements of batches
                if region[1] == 1
                    istride = 1
                    idist = prod(sz)
                    cdist = prod(csz)
                else
                    if region[end] != length(xdims)
                        throw(ArgumentError("batching dims must be sequential"))
                    end
                    istride = prod(xdims[1:region[1]-1])
                    idist = 1
                    cdist = 1
                end
                inembed = Cint[rsz...]
                cnembed = (length(csz) > 1) ? Cint[reverse(csz)...] : Cint[csz[1]]
                ostride = istride
                if xtype == CUFFT_R2C || xtype == CUFFT_D2Z
                    odist = cdist
                    onembed = cnembed
                else
                    odist = idist
                    onembed = inembed
                end
                if xtype == CUFFT_C2R || xtype == CUFFT_Z2D
                    idist = cdist
                    inembed = cnembed
                end
            else
                if any(diff(collect(region)) .< 1)
                    throw(ArgumentError("region must be an increasing sequence"))
                end
                cdims = collect(xdims)
                cdims[region[1]] = div(cdims[region[1]],2)+1

                if region[1] == 1
                    istride = 1
                    ii=1
                    while (ii < nrank) && (region[ii] == region[ii+1]-1)
                        ii += 1
                    end
                    idist = prod(xdims[1:ii])
                    cdist = prod(cdims[1:ii])
                    ngaps = 0
                else
                    istride = prod(xdims[1:region[1]-1])
                    idist = 1
                    cdist = 1
                    ngaps = 1
                end
                nem = ones(Int,nrank)
                cem = ones(Int,nrank)
                id = 1
                for ii=1:nrank-1
                    if region[ii+1] > region[ii]+1
                        ngaps += 1
                    end
                    while id < region[ii+1]
                        nem[ii] *= xdims[id]
                        cem[ii] *= cdims[id]
                        id += 1
                    end
                    @assert nem[ii] >= sz[ii]
                end
                if region[end] < length(xdims)
                    ngaps += 1
                end
                # CUFFT represents batches by a single stride (_dist)
                # so we must verify that region is consistent with this:
                if ngaps > 1
                    throw(ArgumentError("batch regions must be sequential"))
                end

                inembed = Cint[reverse(nem)...]
                cnembed = Cint[reverse(cem)...]
                ostride = istride
                if xtype == CUFFT_R2C || xtype == CUFFT_D2Z
                    odist = cdist
                    onembed = cnembed
                else
                    odist = idist
                    onembed = inembed
                end
                if xtype == CUFFT_C2R || xtype == CUFFT_Z2D
                    idist = cdist
                    inembed = cnembed
                end
            end
            cufftMakePlanMany(handle, nrank, Cint[rsz...],
                              inembed, istride, idist, onembed, ostride, odist,
                              xtype, batch, worksize_ref)
        end
    end

    handle, worksize_ref[]
end

# plan cache
const cufftHandleCacheKey = Tuple{CuContext, cufftType_t, Dims, Any}
const idle_handles = HandleCache{cufftHandleCacheKey, cufftHandle}()
function cufftGetPlan(args...)
    ctx = context()
    handle = pop!(idle_handles, (ctx, args...)) do
        # make the plan
        handle, worksize = cufftMakePlan(args...)

        # NOTE: we currently do not use the worksize to allocate our own workarea,
        #       instead relying on the automatic allocation strategy.
        handle
    end

    # assign to the current stream
    cufftSetStream(handle, stream())

    return handle
end
function cufftReleasePlan(plan)
    push!(idle_handles, plan) do
        cufftDestroy(plan)
    end

end
