### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 3ae6edc8-7046-47ad-8660-b7cab0f848b5
using Pkg

# ╔═╡ ee698f4b-2802-477f-9dbe-707879463ffd
Pkg.activate(".")

# ╔═╡ 04cea513-a856-4a06-8862-b19cd3fe6e04
Pkg.add(url = "https://github.com/MohamedLaghdafHABIBOULLAH/ShiftedProximalOperators.jl", rev = "Rank")

# ╔═╡ a99b68df-ddb2-4534-8eb6-d59e29e8af88
Pkg.add(url = "https://github.com/optimizers/RegularizedProblems.jl")

# ╔═╡ bfdee2eb-b236-4237-80e0-a8134ffbe90b
Pkg.add(url = "https://github.com/MohamedLaghdafHABIBOULLAH/RegularizedOptimization.jl", rev = "R2_edited")

# ╔═╡ d6334a44-903e-415b-9c0b-dd95536a2d04
Pkg.add("Images")

# ╔═╡ beeb6bff-5507-476f-bcff-eddd2fe77a94
Pkg.add("Noise")

# ╔═╡ c77f85ad-484f-417a-8d0c-b31e6ecddd9d
Pkg.add("Distributions")

# ╔═╡ 33d39aac-b99f-41a4-93f5-03cafd0ef8d6
Pkg.add("PlutoUI"); using PlutoUI

# ╔═╡ 9e374209-8fea-4a5e-b4cb-cdddd9c8a1ca
using ShiftedProximalOperators

# ╔═╡ 9a0f3cb9-4642-4ca7-bdda-f27cd3235c1c
using RegularizedProblems

# ╔═╡ 560e183e-da4f-4634-b26c-c18fe3007b6d
using RegularizedOptimization

# ╔═╡ 5aff4797-59f2-41c7-bf8d-40e55bdac326
using Images

# ╔═╡ c14b8791-362e-4bc4-ae58-f64acc74320f
using Noise

# ╔═╡ 96efa3ee-a6d0-4547-af98-265368e89d56
using Distributions

# ╔═╡ 178ce872-279d-4597-aef1-8e856ba8fd57
using LinearAlgebra

# ╔═╡ 5d76bb73-b175-453b-b83a-bd64fd8d84ff
pwd()

# ╔═╡ a6e7e539-d135-4cf2-8436-3320eae17216
mkdir("rank-problem")

# ╔═╡ e25e633e-0cef-4a55-a280-be20e96c9fa2
cd("rank-problem")

# ╔═╡ 5c953ed9-7e3f-47f9-9101-d03e497e3288
# Definir l'image
begin
    I = ones(256,256)
    I[:,1:20] .= 0.
    I[1:126,40:60] .= 0
    I[:,80:100] .= 0
    I[1:40, 120 : 140] .= 0
    I[80:256, 120 : 140] .= 0.5
    I[1: 40, 160:256] .= 0
    I[80:256, 160 : 180] .= 0
end

# ╔═╡ 7476b12a-c613-44bd-863f-50c8103c77d6
Gray.(I)

# ╔═╡ cbf20cac-7388-4e7f-a301-02ee524f7944
# Definir la perturbation
function Perturb(I, c, p)
	Omega = findall(<(p), rand(256,256))
	Omega_vec = zeros(Int64, size(Omega,1))   # Vectorize Omega 
    for i = 1:size(Omega,1)
        Omega_vec[i] = Omega[i][1] + 256 * (Omega[i][2] - 1)
    end
	X = zeros(size(I))
	B = I[Omega]
	B = c*add_gauss(B, sqrt(0.001), 0; clip = true) + (1-c)*add_gauss(B, sqrt(0.1), 0; clip = true)
	X[Omega] = B
	X, B, Omega_vec
end

# ╔═╡ e88e0bee-7d6b-4025-bbc2-1e1ccc60044d
X, B, Omega = Perturb(I, 0.9, 0.8);

# ╔═╡ 3854a17f-ef7e-44b1-bc74-7aba29642cb6
size(B)

# ╔═╡ 10a48ffa-3350-498c-8a8a-197b5c0a7691
Gray.(X)

# ╔═╡ c17b0eaf-fe93-4dac-b83a-d08a1ff3bfae
# Definir le modèle
function model_image(B, Omega, xinit)
    res = similar(xinit)

    function resid!(res, x)
        res .= 0
        res[Omega] .=  x[Omega] .- B
        res 
    end

    function obj(x)
        resid!(res, x)
        dot(res, res) / 2
    end

    grad!(g, x) = resid!(g, x)
    
    function jprod_resid!(Jv, x, v)
        Jv .= 0
        Jv[Omega] .=  v[Omega]
        Jv 
    end

    FirstOrderModel(obj, grad!, xinit, name = "MIT"),
    FirstOrderNLSModel(resid!, jprod_resid!, jprod_resid!, size(xinit,1), xinit, name = "MIT-LS") 
end

# ╔═╡ 9898bfa3-f4c8-4678-b4f5-a38f6e25fb3a
# Definir le point initial
# xinit = vec(reshape(X, 256*256, 1));
xinit = rand(256*256);

# ╔═╡ 265c0691-0036-4fc7-8ed5-01afd72f4668
# Definir le modele lisse
f, g = model_image(B, Omega, xinit);

# ╔═╡ 9dd371ba-23dc-4094-b58e-53d438d8da82
# Definir le modele non lisse
hr = Rank(10., ones(256,256));

# ╔═╡ f86665b0-4205-4db3-9067-a0d9320bee60
# Definir la solution
options_R2 = ROSolverOptions(ν = 1., β = 1e16, ϵ = 1e-3, verbose = 10, maxIter = 10000, σmin = 1.e-10);

# ╔═╡ c50b4b4f-112d-461f-9011-8456b43ff9ca
options_R2.σmin

# ╔═╡ 336b2e81-a4eb-411d-9199-af2bcc1bfefb
options_LM = ROSolverOptions(;ν = 1.0e+2, β = 1e16, ϵ = 1e-3, verbose = 10, σmin = 1e-3);

# ╔═╡ 38b4cd44-01a7-4ce8-8218-fe4796134a57
out_R2 = with_terminal() do
    R2(f, hr, options_R2, x0 = rand(256*256))
end

# ╔═╡ 617abc89-cf17-4600-bc02-a254336e2234
out_LM = with_terminal() do
    LM(g, hr, options_LM)
end

# ╔═╡ 43d7fff6-72ee-4efb-8282-9d84a40528de
# Solution R2
stats_R2 = out_R2.value

# ╔═╡ 1f70e514-f763-43af-84d5-244539809155
# Representer la solution
Gray.(ShiftedProximalOperators.reshape_array(stats_R2.solution, (256,256)))

# ╔═╡ f1518c5e-28de-4a8e-9634-4ca2501907c3
# Rang de la solution
rank(ShiftedProximalOperators.reshape_array(stats_R2.solution, (256,256)))

# ╔═╡ 3f024371-15a3-4f80-9fc6-035963a55559
# Temps
stats_R2.elapsed_time

# ╔═╡ dc508194-1823-4dac-83d4-f2a7e234ab7f
maximum(I)

# ╔═╡ a2662c8b-bb47-4759-be0e-a642c42a19c2
10*log(10,256*256/(norm(ShiftedProximalOperators.reshape_array(stats_R2.solution, (256,256)) - I)^2))

# ╔═╡ 87b9fb1b-52dc-433c-825a-286d4576b74a
# Solution LM
stats_LM = out_LM.value

# ╔═╡ 73442eb1-4cfa-42c6-890a-d2498d7ebf05
norm(ShiftedProximalOperators.reshape_array(stats_LM.solution, (256,256)) - I)/256

# ╔═╡ fb0855d8-dfa4-46dc-bf8a-3ce5c0a6af17
# Representer la solution
Gray.(ShiftedProximalOperators.reshape_array(stats_LM.solution, (256,256)))

# ╔═╡ fb3046db-c930-4e31-bf13-162938f98f50
# Rang de la solution
rank(ShiftedProximalOperators.reshape_array(stats_LM.solution, (256,256)))

# ╔═╡ 83ddd378-b477-4c46-87d6-89543ba84047
# Temps
stats_LM.elapsed_time

# ╔═╡ f3f1f70e-e9f1-4e0b-b0e6-27f219fb1f6e
10*log(10,256*256/(norm(ShiftedProximalOperators.reshape_array(stats_LM.solution, (256,256)) - I)^2))

# ╔═╡ 877a4a72-d4ee-4b04-afc4-4c543b074d85
## Comparer les solutions
norm(ShiftedProximalOperators.reshape_array(stats_R2.solution, (256,256)) - ShiftedProximalOperators.reshape_array(stats_LM.solution, (256,256)))


# ╔═╡ 8618c388-6c46-45dc-b0b1-c4bf761681fa
function mat_rand(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xl = rand(Uniform(-0.1, 0.3), m, r)
  xr = rand(Uniform(-0.1, 0.3), n, r)
  xs = xl * xr'
  Ω = findall(<(sr), rand(m, n))
  B = xs[Ω]
  B = (1 - c) * add_gauss(B, va, 0; clip = true) + c * add_gauss(B, vb, 0; clip = true)
  return xs, B, Ω
end

# ╔═╡ 34874f4c-7588-4b39-bf07-f051a20e2a8e
function mat_rand_model(m::Int, n::Int, r::Int, sr::Float64, va::Float64, vb::Float64, c::Float64)
  xs, B, Ω = mat_rand(m, n, r, sr, va, vb, c)
  res = zeros(m, n)

  function resid!(res, x)
    res .= zeros(m, n)
    res[Ω] .= ShiftedProximalOperators.reshape_array(x, (m, n))[Ω] .- B
    vec(res)
  end

  function obj(x)
    resid!(res, x)
    dot(res, res) / 2
  end

  grad!(r, x) = resid!(ShiftedProximalOperators.reshape_array(r, (m, n)), x)

  return FirstOrderModel(obj, grad!, rand(Float64, m * n), name = "MATRAND"), xs
end


# ╔═╡ c3ea717d-7245-421b-bdc2-97ba636cbc21
# Define model
model, xsol = mat_rand_model(200, 200, 5, 0.8, sqrt(0.0001), sqrt(0.1), 0.2);

# ╔═╡ 5d884c83-75c7-48ad-b794-38489b292f49
hr2 = Rank(1.5, ones(200,200));

# ╔═╡ ea57eb02-cfe1-4111-9a88-4ea27d4466bd
options2 = ROSolverOptions(ν = 1., β = 1e16, ϵ = 1e-3, verbose = 10, maxIter = 300);

# ╔═╡ 582a4760-c8cf-4d34-a98b-61cc70f00d91
x2 = vec(rand(200*200));

# ╔═╡ 952bb2b8-3578-4bac-bdf2-b930e0ea497d
out2_R2 = with_terminal() do
    R2(model, hr2, options2, x0 = x2)
end

# ╔═╡ 91b5f785-7e02-40e4-ac2c-e90d7b418037
# Solution R2
stats2 = out2_R2.value

# ╔═╡ 78a7898f-2817-47d3-9b29-a826dc1cf8ef
stats2.elapsed_time

# ╔═╡ 79059ca4-eea5-4090-996e-4fd2a8cf5d04
rank(reshape(stats2.solution,200,200))

# ╔═╡ 492b6e2c-0399-4640-82a3-740eb1288b65
norm(reshape(xsol,200*200,1) - stats2.solution)/200.

# ╔═╡ f9816be3-5ed1-44b4-ba3f-ac8bb4ee2d1b
maximum(xsol)

# ╔═╡ Cell order:
# ╠═5d76bb73-b175-453b-b83a-bd64fd8d84ff
# ╠═a6e7e539-d135-4cf2-8436-3320eae17216
# ╠═e25e633e-0cef-4a55-a280-be20e96c9fa2
# ╠═3ae6edc8-7046-47ad-8660-b7cab0f848b5
# ╠═ee698f4b-2802-477f-9dbe-707879463ffd
# ╠═04cea513-a856-4a06-8862-b19cd3fe6e04
# ╠═a99b68df-ddb2-4534-8eb6-d59e29e8af88
# ╠═bfdee2eb-b236-4237-80e0-a8134ffbe90b
# ╠═9e374209-8fea-4a5e-b4cb-cdddd9c8a1ca
# ╠═9a0f3cb9-4642-4ca7-bdda-f27cd3235c1c
# ╠═560e183e-da4f-4634-b26c-c18fe3007b6d
# ╠═d6334a44-903e-415b-9c0b-dd95536a2d04
# ╠═5aff4797-59f2-41c7-bf8d-40e55bdac326
# ╠═beeb6bff-5507-476f-bcff-eddd2fe77a94
# ╠═c14b8791-362e-4bc4-ae58-f64acc74320f
# ╠═c77f85ad-484f-417a-8d0c-b31e6ecddd9d
# ╠═96efa3ee-a6d0-4547-af98-265368e89d56
# ╠═178ce872-279d-4597-aef1-8e856ba8fd57
# ╠═33d39aac-b99f-41a4-93f5-03cafd0ef8d6
# ╠═5c953ed9-7e3f-47f9-9101-d03e497e3288
# ╠═7476b12a-c613-44bd-863f-50c8103c77d6
# ╠═cbf20cac-7388-4e7f-a301-02ee524f7944
# ╠═e88e0bee-7d6b-4025-bbc2-1e1ccc60044d
# ╠═3854a17f-ef7e-44b1-bc74-7aba29642cb6
# ╠═10a48ffa-3350-498c-8a8a-197b5c0a7691
# ╠═c17b0eaf-fe93-4dac-b83a-d08a1ff3bfae
# ╠═9898bfa3-f4c8-4678-b4f5-a38f6e25fb3a
# ╠═265c0691-0036-4fc7-8ed5-01afd72f4668
# ╠═9dd371ba-23dc-4094-b58e-53d438d8da82
# ╠═f86665b0-4205-4db3-9067-a0d9320bee60
# ╠═c50b4b4f-112d-461f-9011-8456b43ff9ca
# ╠═336b2e81-a4eb-411d-9199-af2bcc1bfefb
# ╠═38b4cd44-01a7-4ce8-8218-fe4796134a57
# ╠═617abc89-cf17-4600-bc02-a254336e2234
# ╠═43d7fff6-72ee-4efb-8282-9d84a40528de
# ╠═1f70e514-f763-43af-84d5-244539809155
# ╠═f1518c5e-28de-4a8e-9634-4ca2501907c3
# ╠═3f024371-15a3-4f80-9fc6-035963a55559
# ╠═dc508194-1823-4dac-83d4-f2a7e234ab7f
# ╠═a2662c8b-bb47-4759-be0e-a642c42a19c2
# ╠═73442eb1-4cfa-42c6-890a-d2498d7ebf05
# ╠═87b9fb1b-52dc-433c-825a-286d4576b74a
# ╠═fb0855d8-dfa4-46dc-bf8a-3ce5c0a6af17
# ╠═fb3046db-c930-4e31-bf13-162938f98f50
# ╠═83ddd378-b477-4c46-87d6-89543ba84047
# ╠═f3f1f70e-e9f1-4e0b-b0e6-27f219fb1f6e
# ╠═877a4a72-d4ee-4b04-afc4-4c543b074d85
# ╠═8618c388-6c46-45dc-b0b1-c4bf761681fa
# ╠═34874f4c-7588-4b39-bf07-f051a20e2a8e
# ╠═c3ea717d-7245-421b-bdc2-97ba636cbc21
# ╠═5d884c83-75c7-48ad-b794-38489b292f49
# ╠═ea57eb02-cfe1-4111-9a88-4ea27d4466bd
# ╠═582a4760-c8cf-4d34-a98b-61cc70f00d91
# ╠═952bb2b8-3578-4bac-bdf2-b930e0ea497d
# ╠═91b5f785-7e02-40e4-ac2c-e90d7b418037
# ╠═78a7898f-2817-47d3-9b29-a826dc1cf8ef
# ╠═79059ca4-eea5-4090-996e-4fd2a8cf5d04
# ╠═492b6e2c-0399-4640-82a3-740eb1288b65
# ╠═f9816be3-5ed1-44b4-ba3f-ac8bb4ee2d1b
