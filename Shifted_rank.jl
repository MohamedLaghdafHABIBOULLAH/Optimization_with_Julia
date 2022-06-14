using ProximalOperators
using ShiftedProximalOperators
using LinearAlgebra

export ShiftedRank
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



mutable struct ShiftedRank{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
} <: ShiftedProximableFunction
  h::Rank1{R}
  xk::V0  # base shift (nonzero when shifting an already shifted function)
  sj::V1  # current shift
  sol::V2   # internal storage
  shifted_twice::Bool
  function ShiftedRank(
    h::Rank1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    shifted_twice::Bool,
    ) where {R <: Real}
    sol = similar(xk)
    new{R, typeof(xk), typeof(sj), typeof(sol)}(h, xk, sj, sol, shifted_twice)
  end
end

fun_name(ψ::ShiftedRank) = "shifted Rank"
fun_expr(ψ::ShiftedRank) = "t ↦ rank(xk + sj + t)"

shifted(h::Rank1{R}, xk::AbstractVector{R}) where {R <: Real} =
  ShiftedRank(h, xk, zero(xk), false)
shifted(
  ψ::ShiftedRank{R, V0, V1, V2},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}} =
  ShiftedRank(ψ.h, ψ.xk, sj, true)




function prox!(
  y::AbstractVector{R},
  ψ::ShiftedRank{R, V0, V1, V2},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
    XS = reshape(ψ.xk + ψ.sj, ψ.h.nrow, ψ.h.ncol)
    Q = reshape(q + ψ.xk + ψ.sj, ψ.h.nrow, ψ.h.ncol)
    SQ = svd(Q)
    yvec = SQ.S
    c = sqrt(2 * ψ.λ * σ)
    for i ∈ eachindex(SQ.S)
      over = abs(SQ.S[i]) > c
      yvec[i] = over * SQ.S[i]
    end
    y = vec(reshape(SQ.U * Diagonal(yvec) * SQ.Vt - XS, ψ.h.nrow * ψ.h.ncol, 1))
    return y
    end
end


##  Tests 
h = Rank1(5.,2,2)
y = zeros(4)
f1 = ShiftedRank(h,[1.0, 1.0, 8.0, 1.0], [0., 0., 0., 0.], false)

prox!(y,f1,[1.0; 8.0; 8.0; 10.0],5.0) == prox!(y,h,[1.0; 8.0; 8.0; 10.0] + [1.0, 1.0, 8.0, 1.0] ,5.) - [1.0, 1.0, 8.0, 1.0]




