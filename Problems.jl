export lrcomp_model, mat_rand_model


reshape2(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)


function lrcomp_data(m::Int, n::Int)
  A = Array(rand(Float64,(m, n)))
  A
end

function lrcomp_model(m::Int, n::Int)
  A = lrcomp_data(m, n)
  r = vec(similar(A))

  function resid!(r, x, A)
    r .=  x .- vec(reshape2(A, (m*n,1)))
    r
  end


  function obj(x)
    resid!(r, x, A)
    dot(r, r) / 2
  end

  function grad!(r, x)
    resid!(r, x, A) 
    r
  end

  FirstOrderModel(obj, grad!, rand(Float64, m*n), name = "LRCOMP")
end


# Preallouer r dans l'algo ci-haut meme avec vec il sera plus exact.



using Distributions
using Noise

# Ω désigne le sous ensemble des points connus et alea
#xl = Array(rand(Uniform(-0.3,0.1), 30, 5))
#xr = Array(rand(Uniform(-0.3,0.1), 30, 5))
#xs = xl*xr'
function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = Array(rand(Uniform(-0.3,0.1),m, r))
  xr = Array(rand(Uniform(-0.3,0.1),n, r))
  xs = xl*xr'
  Ω = findall(<(sr), rand(m,n));
  B=xs[Ω];
  B = (1-c)*add_gauss(B, va, 0) + c*add_gauss(B, vb, 0);
  return xs, B, Ω
end



function mat_rand_model(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  T = mat_rand(m, n, r, sr, va, vb, c)
  res = zeros(m,n)

  function resid!(res, x)
    res[T[3]] .= vec(T[2] - reshape2(x, (m,n))[T[3]])
    vec(res)
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  function grad!(g, x)
    resid!(res,x)
    res
  end
  function REL(x)
    rel=sqrt(norm(x-reshape2(T[1],(m*n,1)))/(m*n));
    rel
  end

  return FirstOrderModel(obj, grad!, rand(Float64, m*n), name = "LRCOMP"), REL, T[1] # T otain xs and compute RMSE
end
