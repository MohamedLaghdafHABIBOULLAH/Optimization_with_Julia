"""
    Rank1(λ)
Return the rank
```math
f(x) = λ*rank(matrix(x)),
```
where `λ` is a positive parameter and x is a vector.
"""

export Rank1
struct Rank1{R<: Real}
    lambda::R
    nrow::Int
    ncol::Int
    function Rank1(lambda::R, nrow::Int, ncol::Int) where {R <: Real}
        if lambda < 0 || nrow <= 0 || ncol <= 0
            error("parameters λ, nrow and ncol must be nonnegative")
        end
        new{typeof(lambda)}(lambda, nrow, ncol)
    end
end

Rank1(lambda::R, nrow::Int, ncol::Int) where {R} =  Rank1{R}(lambda, nrow, ncol)

function (f::Rank1)(x::AbstractVector{R}) where {R <: Real}
    return f.lambda * rank(reshape(x, f.nrow, f.ncol))
end


function prox!(y::AbstractVector{R}, f::Rank1{R}, x::AbstractVector{R}, gamma::R) where {R <: Real}
    A = reshape(x, f.nrow, f.ncol)
    F = svd(A)
    y = vec(reshape(F.U * Diagonal(ProximalOperators.prox_naive(NormL0(f.lambda),F.S,gamma)[1]) * F.Vt, f.nrow * f.ncol, 1))
    return y
end
