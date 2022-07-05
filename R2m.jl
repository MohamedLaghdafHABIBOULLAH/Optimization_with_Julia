using LinearAlgebra, Logging, Printf

using RegularizedProblems
# external dependencies
using Arpack, ProximalOperators

# dependencies from us
using NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore


export R2a, PGa


function R2m(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  xk, k, outdict = R2m(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)

  return GenericExecutionStats(
    outdict[:status],
    nlp,
    solution = xk,
    objective = outdict[:fk] + outdict[:hk],
    dual_feas = sqrt(outdict[:ξ]),
    iter = k,
    elapsed_time = outdict[:elapsed_time],
    solver_specific = Dict(
      :Fhist => outdict[:Fhist],
      :Hhist => outdict[:Hhist],
      :NonSmooth => outdict[:NonSmooth],
      :SubsolverCounter => outdict[:Chist],
    ),
  )
end

function R2m(
  f::F,
  ∇f!::G,
  h::H,
  options::ROSolverOptions,
  x0::AbstractVector,
) where {F <: Function, G <: Function, H}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵ
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  η1 = options.η1
  η2 = options.η2
  ν = options.ν
  γ = options.γ

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  xk = copy(x0)
  hk = h(xk)
  if hk == Inf
    verbose > 0 && @info "R2: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk)
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "R2: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")

  xkn = similar(xk)
  s = zero(xk)
  ψ = shifted(h, xk)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  verbose == 0 ||
    @info @sprintf "%6s %8s %8s %7s %8s %7s %7s %7s %1s" "iter" "f(x)" "h(x)" "√ξ" "ρ" "σ" "‖x‖" "‖s‖" ""

  local ξ
  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  mν∇fk = -ν * ∇fk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # define model
    φk(d) = dot(∇fk, d)
    mk(d) = φk(d) + ψ(d)

    prox!(s, ψ, mν∇fk, ν)
    Complex_hist[k] += 1
    mks = mk(s)
    ξ = hk - mks + max(1, abs(hk)) * 10 * eps()
    ξ > 0 || error("R2: prox-gradient step should produce a decrease but ξ = $(ξ)")

    if sqrt(ξ) < ϵ
      optimal = true
      continue
    end

    xkn .= xk .+ s
    fkn = f(xkn)
    hkn = h(xkn)
    hkn == -Inf && error("nonsmooth term is not proper")

    Δobj = (fk + hk) - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ρk = Δobj / ξ

    σ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %1s" k fk hk sqrt(ξ) ρk σk norm(xk) norm(s) σ_stat
    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if η1 ≤ ρk < Inf
      xk .= xkn
      fk = fkn
      hk = hkn
      ∇f!(∇fk, xk)
      shift!(ψ, xk)
    end

    if ρk < η1 || ρk == Inf
      σk = σk * γ
    end

    ν = 1 / σk
    tired = maxIter > 0 && k ≥ maxIter
    if !tired
      @. mν∇fk = -ν * ∇fk
    end
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
    elseif optimal
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8s %7.1e %7.1e %7.1e" k fk hk sqrt(ξ) "" σk norm(xk) norm(s)
      @info  "R2: terminating with √ξ = $(sqrt(ξ))"
    end
  end

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end
  outdict = Dict(
    :Fhist => Fobj_hist[1:k],
    :Hhist => Hobj_hist[1:k],
    :Chist => Complex_hist[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end

function PGa(nlp::AbstractNLPModel, args...; kwargs...)
    kwargs_dict = Dict(kwargs...)
    x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
    xk, k, outdict = PGa(x -> obj(nlp, x), (g, x) -> grad!(nlp, x, g), args..., x0; kwargs_dict...)
  
    return GenericExecutionStats(
    outdict[:status],
    nlp,
    solution = xk,
    objective = outdict[:fk] + outdict[:hk],
    dual_feas = sqrt(outdict[:ξ]),
    iter = k,
    elapsed_time = outdict[:elapsed_time],
    solver_specific = Dict(
      :Fhist => outdict[:Fhist],
      :Hhist => outdict[:Hhist],
      :NonSmooth => outdict[:NonSmooth],
      :SubsolverCounter => outdict[:Chist],
    ),
  )
  end
  
  function PGa(
    f::F,
    ∇f!::G,
    h::H,
    options::ROSolverOptions,
    x0::AbstractVector,
  ) where {F <: Function, G <: Function, H}
    start_time = time()
    elapsed_time = 0.0
    ϵ = options.ϵ
    maxIter = options.maxIter
    maxTime = options.maxTime
    ν = options.ν
    verbose = options.verbose
  
    if options.verbose == 0
      ptf = Inf
    elseif options.verbose == 1
      ptf = round(maxIter / 10)
    elseif options.verbose == 2
      ptf = round(maxIter / 100)
    else
      ptf = 1
    end
  
    #Problem Initialize
    xk = copy(x0)
    Fobj_hist = zeros(maxIter)
    Hobj_hist = zeros(maxIter)
    Complex_hist = zeros(Int, maxIter)
  
    # Iteration set up
    ∇fk = similar(xk)
    ∇f!(∇fk, xk)
    fk = f(xk)
    hk = h(xk)
    ∇fkn = similar(∇fk)
    xkn = similar(x0)
    fstep = xk .- ν .* ∇fk
  
    #do iterations
    local ξ
    k = 0
    optimal = false
    tired = k ≥ maxIter || elapsed_time ≥ maxTime
  
    if options.verbose != 0
      @info @sprintf "%6s %8s %8s %7s %8s %7s" "iter" "f(x)" "h(x)" "‖∂ϕ‖" "ν" "‖x‖"
    end
  
    while !(optimal || tired)
      k = k + 1
      elapsed_time = time() - start_time
      Fobj_hist[k] = fk
      Hobj_hist[k] = hk
      Complex_hist[k] += 1
  
      ∇fkn .= ∇fk
      xkn .= xk
      fstep .= xk .- ν .* ∇fk
      prox!(xk, h, fstep, ν)
  
      ∇f!(∇fk, xk)
      fk = f(xk)
      hk = h(xk)
  
      k += 1
      ∇fkn .= ∇fk .- ∇fkn .- (xk .- xkn) ./ ν
      ξ = norm(∇fkn)
      optimal = ξ < ϵ
      tired = k ≥ maxIter || elapsed_time > maxTime
  
      if (verbose > 0) && (k % ptf == 0)
        @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e " k fk hk ξ ν norm(xk)
      end
    end
    status = if optimal
      :first_order
    elseif elapsed_time > maxTime
      :max_time
    elseif tired
      :max_iter
    else
      :exception
    end
  
    outdict = Dict(
      :Fhist => Fobj_hist[1:k],
      :Hhist => Hobj_hist[1:k],
      :Chist => Complex_hist[1:k],
      :NonSmooth => h,
      :status => status,
      :fk => fk,
      :hk => hk,
      :ξ => ξ,
      :elapsed_time => elapsed_time,
    )
  
    return xk, k, outdict
  end
  
