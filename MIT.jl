### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 3ae6edc8-7046-47ad-8660-b7cab0f848b5
using Pkg

# ╔═╡ 04cea513-a856-4a06-8862-b19cd3fe6e04
Pkg.add(url = "https://github.com/MohamedLaghdafHABIBOULLAH/ShiftedProximalOperators.jl", rev = "Rank")

# ╔═╡ a99b68df-ddb2-4534-8eb6-d59e29e8af88
Pkg.add(url = "https://github.com/optimizers/RegularizedProblems.jl")

# ╔═╡ bfdee2eb-b236-4237-80e0-a8134ffbe90b
Pkg.add(url = "https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl")

# ╔═╡ d6334a44-903e-415b-9c0b-dd95536a2d04
Pkg.add("Images")

# ╔═╡ beeb6bff-5507-476f-bcff-eddd2fe77a94
Pkg.add("Noise")

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

# ╔═╡ 178ce872-279d-4597-aef1-8e856ba8fd57
using LinearAlgebra

# ╔═╡ 5d76bb73-b175-453b-b83a-bd64fd8d84ff
pwd()

# ╔═╡ 5c953ed9-7e3f-47f9-9101-d03e497e3288
# Definir l'image
begin
    I = ones(256,256)
    I[:,1:20] .= 0
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
	Omega = findall(<(p), rand(256,256));
	X = zeros(size(I));
	B = I[Omega];
	B = c*add_gauss(B, sqrt(0.001), 0) + (1-c)*add_gauss(B, sqrt(0.1), 0);
	X[Omega] = B;
	X, B, Omega
end

# ╔═╡ e88e0bee-7d6b-4025-bbc2-1e1ccc60044d
X, B, Omega = Perturb(I, 0.8, 0.8);

# ╔═╡ 10a48ffa-3350-498c-8a8a-197b5c0a7691
Gray.(X)

# ╔═╡ c17b0eaf-fe93-4dac-b83a-d08a1ff3bfae
# Definir le modèle
function model_image(B, Omega, xinit)
    res = similar(xinit)
	m, n = size(xinit)

    function resid!(res, x)
        res .= 0
        res[Omega] .= ShiftedProximalOperators.reshape_array(x, (m,n))[Omega] .- B
        vec(res)
    end

    function obj(x)
        resid!(res, x)
        dot(res, res) / 2
    end

    grad!(g, x) = resid!(ShiftedProximalOperators.reshape_array(g, (m, n)), x)

    return FirstOrderModel(obj, grad!, 		 
    vec(ShiftedProximalOperators.reshape_array(xinit, (m*n,1))),name = "MIT") 
end

# ╔═╡ 9898bfa3-f4c8-4678-b4f5-a38f6e25fb3a
# Definir le point initial
# xinit = vec(reshape(X, 256*256, 1));
xinit = rand(256,256);

# ╔═╡ 265c0691-0036-4fc7-8ed5-01afd72f4668
# Definir le modele lisse
f = model_image(B,Omega,xinit);

# ╔═╡ 64caeec2-dd94-4dcd-bee1-f6a2cd91b5d4
Fr = psvd_workspace_dd(zeros(256,256), full=false);

# ╔═╡ 9dd371ba-23dc-4094-b58e-53d438d8da82
# Definir le modele non lisse
hr = Rank(100., ones(256,256), Fr);

# ╔═╡ f86665b0-4205-4db3-9067-a0d9320bee60
# Definir la solution
options = ROSolverOptions(ν = 1., β = 1e16, ϵ = 1e-6, verbose = 10, maxIter = 1000);

# ╔═╡ 20abafc5-40ba-44b4-a35a-704abf671660
# Resoudre avec R2
sol = R2(f, hr, options, x0 = rand(256*256))

# ╔═╡ 1f70e514-f763-43af-84d5-244539809155
# Representer la solution
Gray.(ShiftedProximalOperators.reshape_array(sol.solution, (256,256)))

# ╔═╡ f1518c5e-28de-4a8e-9634-4ca2501907c3
# Rang de la solution
rank(ShiftedProximalOperators.reshape_array(sol.solution, (256,256)))

# ╔═╡ 3f024371-15a3-4f80-9fc6-035963a55559
sol.elapsed_time

# ╔═╡ Cell order:
# ╠═5d76bb73-b175-453b-b83a-bd64fd8d84ff
# ╠═3ae6edc8-7046-47ad-8660-b7cab0f848b5
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
# ╠═178ce872-279d-4597-aef1-8e856ba8fd57
# ╠═5c953ed9-7e3f-47f9-9101-d03e497e3288
# ╠═7476b12a-c613-44bd-863f-50c8103c77d6
# ╠═cbf20cac-7388-4e7f-a301-02ee524f7944
# ╠═e88e0bee-7d6b-4025-bbc2-1e1ccc60044d
# ╠═10a48ffa-3350-498c-8a8a-197b5c0a7691
# ╠═c17b0eaf-fe93-4dac-b83a-d08a1ff3bfae
# ╠═9898bfa3-f4c8-4678-b4f5-a38f6e25fb3a
# ╠═265c0691-0036-4fc7-8ed5-01afd72f4668
# ╠═64caeec2-dd94-4dcd-bee1-f6a2cd91b5d4
# ╠═9dd371ba-23dc-4094-b58e-53d438d8da82
# ╠═f86665b0-4205-4db3-9067-a0d9320bee60
# ╠═20abafc5-40ba-44b4-a35a-704abf671660
# ╠═1f70e514-f763-43af-84d5-244539809155
# ╠═f1518c5e-28de-4a8e-9634-4ca2501907c3
# ╠═3f024371-15a3-4f80-9fc6-035963a55559
