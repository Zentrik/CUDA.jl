# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul
export _spadjoint, _sptranspose

function _spadjoint(A::CuSparseMatrixCSR)
    Aᴴ = CuSparseMatrixCSC(A.rowPtr, A.colVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSR(Aᴴ)
end
function _sptranspose(A::CuSparseMatrixCSR)
    Aᵀ = CuSparseMatrixCSC(A.rowPtr, A.colVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSR(Aᵀ)
end
function _spadjoint(A::CuSparseMatrixCSC)
    Aᴴ = CuSparseMatrixCSR(A.colPtr, A.rowVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSC(Aᴴ)
end
function _sptranspose(A::CuSparseMatrixCSC)
    Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSC(Aᵀ)
end
function _spadjoint(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, conj(A.nzVal), reverse(size(A))..., fmt = :coo)
end
function _sptranspose(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, A.nzVal, reverse(size(A))..., fmt = :coo)
end

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    n_A, m_A = (transa != 'N') ? reverse(size(A)) : size(A)
    n_B, m_B = (transb != 'N') ? reverse(size(B)) : size(B)
    n_C, m_C = size(C)
    m_A == n_B || throw(DimensionMismatch())
    n_A == n_C || throw(DimensionMismatch())
    m_B == m_C || throw(DimensionMismatch())
    isempty(B) && return CUDA.zeros(eltype(B), size(A, 1), 0)
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasReal} = vv!('N', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasReal} = dot(y, x)

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasComplex} = vv!('C', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasComplex} = conj(dot(y,x))

adjtrans_wrappers = ((identity, identity),
                     (M -> :(Transpose{T, <:$M}), M -> :(_sptranspose(parent($M)))),
                     (M -> :(Adjoint{T, <:$M}), M -> :(_spadjoint(parent($M)))))

op_wrappers = ((identity, T -> 'N', identity),
               (T -> :(Transpose{T, <:$T}), T -> 'T', A -> :(parent($A))),
               (T -> :(Adjoint{T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A))),
               (T -> :(HermOrSym{T, <:$T}), T -> 'N', A -> :(parent($A))))

function LinearAlgebra.generic_matvecmul!(C::CuVector{T}, tA::AbstractChar, A::CuSparseMatrix{T}, B::DenseCuVector{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    mv_wrapper(tA, _add.alpha, A, B, _add.beta, C)
end
function LinearAlgebra.generic_matvecmul!(C::CuVector{T}, tA::AbstractChar, A::CuSparseMatrix{T}, B::CuSparseVector{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    mv_wrapper(tA, _add.alpha, A, CuVector{T}(B), _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::CuMatrix{T}, tA, tB, A::CuSparseMatrix{T}, B::DenseCuMatrix{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm_wrapper(tA, tB, _add.alpha, A, B, _add.beta, C)
end

for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(CuSparseMatrix{T}))

    @eval function LinearAlgebra.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: Union{Float16, ComplexF16, BlasFloat}}
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = CuVector{T}(undef, m)
        mul!(y, A, x, true, false)
    end
end

function LinearAlgebra.generic_matvecmul!(C::CuVector{T}, tA::AbstractChar, A::DenseCuMatrix{T}, B::CuSparseVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    gemvi!(tA, _add.alpha, A, B, _add.beta, C, 'O')
end

function LinearAlgebra.generic_matmatmul!(C::CuMatrix{T}, tA, tB, A::DenseCuMatrix{T}, B::CuSparseMatrixCSC{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{T}, tA, tB, A::DenseCuMatrix{T}, B::CuSparseMatrixCSR{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuMatrix{T}, tA, tB, A::DenseCuMatrix{T}, B::CuSparseMatrixCOO{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    mm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end

for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(DenseCuMatrix{T}))

    @eval function Base.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: BlasFloat}
        m, n = size(A)
        length(x) == n || throw(DimensionMismatch())
        y = CuVector{T}(undef, m)
        mul!(y, A, x, true, false)
    end

    for (wrapb, transb, unwrapb) in op_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
            TypeB = wrapb(SparseMatrixType)

            @eval function Base.:(*)(A::$TypeA, B::$TypeB) where {T <: Union{Float16, ComplexF16, BlasFloat}}
                m, n = size(A)
                k, p = size(B)
                n == k || throw(DimensionMismatch())
                C = CuMatrix{T}(undef, m, p)
                mul!(C, A, B, true, false)
            end
        end
    end
end

if VERSION < v"1.10-"
for (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(:(DenseCuMatrix{T}))

    @eval function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::CuSparseVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
        gemvi!($transa(T), alpha, $(unwrapa(:A)), B, beta, C, 'O')
    end
end
end

function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSC{T}, tA, tB, A::CuSparseMatrixCSC{T}, B::CuSparseMatrixCSC{T}, _add::MulAddMul) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    gemm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCSR{T}, tA, tB, A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}, _add::MulAddMul) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    gemm!(tA, tB, _add.alpha, A, B, _add.beta, C, 'O')
end
function LinearAlgebra.generic_matmatmul!(C::CuSparseMatrixCOO{T}, tA, tB, A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}, _add::MulAddMul) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    C_csr = CuSparseMatrixCSR(C)
    generic_matmatmul!(C_csr, tA, tB, A_csr, B_csr, _add.alpha, _add.beta)
    C = CuSparseMatrixCOO(C_csr) # is this in-place of the original C?
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval function LinearAlgebra.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T <: BlasFloat}
        CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
        gemm('N', 'N', one(T), A, B, 'O')
    end
end

function LinearAlgebra.:(*)(A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    CuSparseMatrixCOO(A_csr * B_csr)
end

for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        TypeB = wrapb(SparseMatrixType)
        wrapa == identity && wrapb == identity && continue
        @eval Base.:(*)(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = $(unwrapa(:A)) * $(unwrapb(:B))
    end
end

for op in (:(+), :(-))
    for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}))
            TypeA = wrapa(SparseMatrixType)
            TypeB = wrapb(SparseMatrixType)
            @eval Base.$op(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = geam(one(T), $(unwrapa(:A)), $(op)(one(T)), $(unwrapb(:B)), 'O')
        end
    end

    @eval begin
        Base.$op(A::CuSparseVector{T}, B::CuSparseVector{T}) where {T <: BlasFloat} = axpby(one(T), A, $(op)(one(T)), B, 'O')
        Base.$op(A::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}},
                 B::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}) where {T <: BlasFloat} =
            CuSparseMatrixCOO($(op)(CuSparseMatrixCSR(A), CuSparseMatrixCSR(B)))
    end
end

# triangular
for SparseMatrixType in (:CuSparseMatrixBSR,)

    ## direct
    for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                                (:UnitLowerTriangular, 'L', 'U'),
                                (:UpperTriangular, 'U', 'N'),
                                (:UnitUpperTriangular, 'U', 'U'))
        @eval begin
            # Left division with vectors
            LinearAlgebra.ldiv!(A::$t{T,<:$SparseMatrixType},
                                B::DenseCuVector{T}) where {T<:BlasFloat} =
                sv2!('N', $uploc, $isunitc, one(T), parent(A), B, 'O')

            # Left division with matrices
            LinearAlgebra.ldiv!(A::$t{T,<:$SparseMatrixType},
                                B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                sm2!('N', 'N', $uploc, $isunitc, one(T), parent(A), B, 'O')
        end
    end

    ## adjoint/transpose ('uploc' reversed)
    for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                                (:UnitLowerTriangular, 'U', 'U'),
                                (:UpperTriangular, 'L', 'N'),
                                (:UnitUpperTriangular, 'L', 'U'))

        for (opa, transa) in ((:Transpose, 'T'),
                              (:Adjoint, 'C'))
            @eval begin
                # Left division with vectors
                LinearAlgebra.ldiv!(A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuVector{T}) where {T<:BlasFloat} =
                    sv2!($transa, $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

                # Left division with matrices
                LinearAlgebra.ldiv!(A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                    sm2!($transa, 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')
            end
        end
    end
end

for SparseMatrixType in (:CuSparseMatrixCOO, :CuSparseMatrixCSR, :CuSparseMatrixCSC)

    ## direct
    for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                                (:UnitLowerTriangular, 'L', 'U'),
                                (:UpperTriangular, 'U', 'N'),
                                (:UnitUpperTriangular, 'U', 'U'))
        @eval begin
            # Left division with vectors
            function LinearAlgebra.ldiv!(C::DenseCuVector{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::DenseCuVector{T}) where {T<:BlasFloat}
                if CUSPARSE.version() ≥ v"12.0"
                    sv!('N', $uploc, $isunitc, one(T), parent(A), B, C, 'O')
                else
                    $SparseMatrixType == CuSparseMatrixCOO && throw(ErrorException("This operation is not supported by the current CUDA version."))
                    copyto!(C, B)
                    sv2!('N', $uploc, $isunitc, one(T), parent(A), C, 'O')
                end
            end

            # Left division with matrices
            function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::DenseCuMatrix{T}) where {T<:BlasFloat}
                if CUSPARSE.version() ≥ v"12.0"
                    sm!('N', 'N', $uploc, $isunitc, one(T), parent(A), B, C, 'O')
                else
                    $SparseMatrixType == CuSparseMatrixCOO && throw(ErrorException("This operation is not supported by the current CUDA version."))
                    copyto!(C, B)
                    sm2!('N', 'N', $uploc, $isunitc, one(T), parent(A), C, 'O')
                end
            end

            function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat}
                CUSPARSE.version() < v"12.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
                sm!('N', 'T', $uploc, $isunitc, one(T), parent(A), parent(B), C, 'O')
            end

            # transb = 'C' is not supported.
            function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasReal}
                CUSPARSE.version() < v"12.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
                sm!('N', 'T', $uploc, $isunitc, one(T), parent(A), parent(B), C, 'O')
            end

            function LinearAlgebra.:(\)(A::$t{T,<:$SparseMatrixType}, B::DenseCuVector{T}) where {T}
                m = length(B)
                C = CuVector{T}(undef, m)
                LinearAlgebra.ldiv!(C, A, B)
            end
        end

        for rhs in (:(DenseCuMatrix{T}), :(Transpose{T,<:DenseCuMatrix}), :(Adjoint{T,<:DenseCuMatrix}))
            @eval begin
                function LinearAlgebra.:(\)(A::$t{T,<:$SparseMatrixType}, B::$rhs) where {T}
                    m, n = size(B)
                    C = CuMatrix{T}(undef, m, n)
                    LinearAlgebra.ldiv!(C, A, B)
                end
            end
        end
    end

    ## adjoint/transpose ('uploc' reversed)
    for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                                (:UnitLowerTriangular, 'U', 'U'),
                                (:UpperTriangular, 'L', 'N'),
                                (:UnitUpperTriangular, 'L', 'U'))

        for (opa, transa) in ((:Transpose, 'T'),
                              (:Adjoint, 'C'))
            @eval begin
                # Left division with vectors
                function LinearAlgebra.ldiv!(C::DenseCuVector{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuVector{T}) where {T<:BlasFloat}
                    if CUSPARSE.version() ≥ v"12.0"
                        sv!($transa, $uploc, $isunitc, one(T), parent(parent(A)), B, C, 'O')
                    else
                        copyto!(C, B)
                        sv2!($transa, $uploc, $isunitc, one(T), parent(parent(A)), C, 'O')
                    end
                end

                # Left division with matrices
                function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuMatrix{T}) where {T<:BlasFloat}
                    if CUSPARSE.version() ≥ v"12.0"
                        sm!($transa, 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, C, 'O')
                    else
                        copyto!(C, B)
                        sm2!($transa, 'N', $uploc, $isunitc, one(T), parent(parent(A)), C, 'O')
                    end
                end

                function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat}
                    CUSPARSE.version() < v"12.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
                    sm!($transa, 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), C, 'O')
                end

                # transb = 'C' is not supported.
                function LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasReal}
                    CUSPARSE.version() < v"12.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
                    sm!($transa, 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), C, 'O')
                end

                function LinearAlgebra.:(\)(A::$t{T,<:$opa{T,<:$SparseMatrixType}}, B::DenseCuVector{T}) where {T}
                    m = length(B)
                    C = CuVector{T}(undef, m)
                    LinearAlgebra.ldiv!(C, A, B)
                end
            end

            for rhs in (:(DenseCuMatrix{T}), :(Transpose{T,<:DenseCuMatrix}), :(Adjoint{T,<:DenseCuMatrix}))
                @eval begin
                    function LinearAlgebra.:(\)(A::$t{T,<:$opa{T,<:$SparseMatrixType}}, B::$rhs) where {T}
                        m, n = size(B)
                        C = CuMatrix{T}(undef, m, n)
                        LinearAlgebra.ldiv!(C, A, B)
                    end
                end
            end
        end
    end
end

## uniform scaling

# these operations materialize the identity matrix and re-use broadcast
# TODO: can we do without this, and just use the broadcast implementation
#       with a singleton argument it knows how to index?

function _sparse_identity(::Type{<:CuSparseMatrixCSR{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    rowPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[1]-len+1)))
    colVal = CuVector{Ti}(1:len)
    nzVal = CUDA.fill(I.λ, len)
    CuSparseMatrixCSR{Tv,Ti}(rowPtr, colVal, nzVal, dims)
end

function _sparse_identity(::Type{<:CuSparseMatrixCSC{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    colPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[2]-len+1)))
    rowVal = CuVector{Ti}(1:len)
    nzVal = CUDA.fill(I.λ, len)
    CuSparseMatrixCSC{Tv,Ti}(colPtr, rowVal, nzVal, dims)
end

function _sparse_identity(::Type{<:CuSparseMatrixCOO{Tv,Ti}},
                        I::UniformScaling, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    rowInd = CuVector{Ti}(1:len)
    colInd = CuVector{Ti}(1:len)
    nzVal = CUDA.fill(I.λ, len)
    CuSparseMatrixCOO{Tv,Ti}(rowInd, colInd, nzVal, dims)
end

for (wrapa, unwrapa) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        @eval begin
            Base.:(+)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) + _sparse_identity(typeof(A), J, size(A))
            Base.:(+)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) + $(unwrapa(:A))

            Base.:(-)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) - _sparse_identity(typeof(A), J, size(A))
            Base.:(-)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) - $(unwrapa(:A))
        end

        # Broadcasting is not yet supported for COO matrices
        if SparseMatrixType != :(CuSparseMatrixCOO{T})
            @eval begin
                Base.:(*)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) * J.λ
                Base.:(*)(J::UniformScaling, A::$TypeA) where {T} = J.λ * $(unwrapa(:A))
            end
        else
            @eval begin
                Base.:(*)(A::$TypeA, J::UniformScaling) where {T} = $(unwrapa(:A)) * _sparse_identity(typeof(A), J, size(A))
                Base.:(*)(J::UniformScaling, A::$TypeA) where {T} = _sparse_identity(typeof(A), J, size(A)) * $(unwrapa(:A))
            end
        end
    end
end

# TODO: let Broadcast handle this automatically (a la SparseArrays.PromoteToSparse)
for (wrapa, unwrapa) in adjtrans_wrappers, op in (:(+), :(-), :(*))
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        @eval begin
            function Base.$op(lhs::Diagonal, rhs::$TypeA) where {T}
                return $op($SparseMatrixType(lhs), $(unwrapa(:rhs)))
            end
            function Base.$op(lhs::$TypeA, rhs::Diagonal) where {T}
                return $op($(unwrapa(:lhs)), $SparseMatrixType(rhs))
            end
        end
    end
end
