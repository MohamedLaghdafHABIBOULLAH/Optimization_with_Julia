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


## Objective function of the prox
function f_obj(t::AbstractVector{R}, ψ::ShiftedRank{R, V0, V1, V2}, q::AbstractVector{R}, σ::R) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}}
  T = reshape(t, ψ.h.nrow, ψ.h.ncol)
  Q = reshape(q, ψ.h.nrow, ψ.h.ncol)
  XS = reshape(ψ.xk + ψ.sj, ψ.h.nrow, ψ.h.ncol)
  return 0.5 * norm(T - Q)^2 + ψ.λ * σ * rank(XS + T)
end

### Squared matrix (10*10)

A1 = Rot(pi/4)
A2 = Rot(pi/6)
A3 = Rot(pi)
A4 = Rot(pi/3)
A5 = Rot(pi/2)
O = vcat([A1 zeros(2,8)], [zeros(2,2) A2 zeros(2,6)], [zeros(2,4) A3 zeros(2,4)], [zeros(2,6) A4 zeros(2,2)], [zeros(2,8) A5] )
Atest = O * Diagonal([8., 5.5, 3., 2.5, 2., 1., 0., 0., 0., 0.]) * transpose(O)

h2 = Rank1(10.,10,10)
y2 = zeros(100)
xtest2 = vec(reshape(Atest,100,1))
f2 = ShiftedRank(h2, xtest, zeros(100), false)

norm(prox!(y2, f2, xtest2, 100.0) - (prox!(y2, h2, 2*xtest2 ,100.) - xtest2 )) <= 1e-11 

f_obj(prox!(y2, f2, xtest, 100.0), f2, xtest, 100.)  ### Evaluating it

### Rectangular matrix (10*11)

Btest = O * Diagonal([8., 5.5, 3., 2.5, 2., 1., 0., 0., 0., 0.]) * hcat(transpose(O), ones(10))
xtest3 = vec(reshape(Btest,110,1))
stest3 = vec(reshape(O*Btest,110,1))
qtest3 = vec(reshape(O^2 *Btest,110,1))
                
h3 = Rank1(10.,10,11)
f3 = ShiftedRank(h3,xtest3, stest3, true)
y3 = zeros(110)

norm(prox!(y3, f3, qtest3, 100.0) - (prox!(y3, h3, qtest3 + xtest3 + stest3, 100.) - xtest3 - stest3)) <= 1e-11 


### Random Rectangular matrix (10*11)

xtest4 = vec(reshape(rand(10,11),110,1))
qtest4 = vec(reshape(rand(10,11),110,1))
stest4 = vec(reshape(rand(10,11),110,1))
h4 = Rank1(10.,10,11)
f4 = ShiftedRank(h2,xtest4, stest4, true)
y4 = zeros(110)

norm(prox!(y4, f4, qtest4, 10.0) - (prox!(y4, h4, qtest4 + xtest4 + stest4 ,10.) - xtest4 - stest4)) <= 1e-11 


##### Diagonal matrix (10*10)




st1 = rand(10)

xtest5 = vec(reshape(Diagonal(st1),100,1))
qtest5 = vec(reshape(Diagonal(st1.^2),100,1))
stest5 = vec(reshape(Diagonal(st1/2),100,1))

h5 = Rank1(10.,10,10)
f5 = ShiftedRank(h5,xtest5, stest5, true)
y5 = zeros(100)


k5 = NormL0(10.)

t5 = ones(10)
using ProximalOperators

ProximalOperators.prox!(t5,k5, st1 + st1.^2 + st1/2, 10.0)

norm(Diagonal(t5 - st1 - st1/2) - reshape(prox!(y5, f5, qtest5, 10.0),10,10)) <= 1e-12



