"""
    Rank1(λ)
Return the rank
```math
f(x) = λ*rank(matrix(x)),
```
where `λ` is a positive parameter and x is a vector.

"""


export Rank1

struct Rank1{R<: Real, V <: Int64}
    lambda::R
    m::V
    n::V
    function Rank1(lambda::R, m::V, n::V) where {R <: Real, V<: Int64}
        if lambda < 0 || n <= 0 || m <= 0
            error("parameter λ,n,m must be nonnegative")
        else
            new{typeof(lambda),typeof(m)}(lambda,m,n)
        end
    end
end

Rank1(lambda::R, m::V, n::V) where {R,V} =  Rank1{R,V}(lambda,m,n)

function (f::Rank1)(x::AbstractVector{R}) where {R <: Real}
    return f.lambda*rank(reshape(x,f.m,f.n))
end


function prox!(y::AbstractVector{R}, f::Rank1{R,V}, x::AbstractVector{R}, gamma::R) where {R <: Real, V<: Int}
    A = reshape(x,f.m,f.n)
    F = svd(A)
    y = vec(reshape(F.U * Diagonal(ProximalOperators.prox_naive(NormL0(f.lambda),F.S,gamma)[1]) * F.Vt, f.m*f.n, 1))
    return y
end
