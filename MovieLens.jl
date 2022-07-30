### read MovieLens data

# include("../../data")


using DataFrames
using DelimitedFiles

data = readdlm("Movi_train (1).csv", ',', Float64)
data_test = readdlm("Movi_test.csv", ',', Float64)
data[1,1] = 5.


using ShiftedProximalOperators
using RegularizedOptimization
using RegularizedProblems
using LinearAlgebra

Omega = findall(>(0.), data);

function convertomega(Omega)
    Om = zeros(Int64, size(Omega,1))
    for i = 1:Int.(size(Omega,1))
        Om[i] = Omega[i][1] + 943 * (Omega[i][2] - 1)
    end
    Om
end

Om = convertomega(Omega)

function mat_rand_model3(m::Int, n::Int, B, Omega1, xinit)
    res = zeros(m*n)

    function resid!(res, x)
        res .= 0
        res[Omega1] .=  x[Omega1] .- B
        res 
    end

    function obj(x)
        resid!(res, x)
        dot(res, res) / 2
    end

    grad!(g, x) = resid!(g, x)
    
    function jprod_resid!(Jv, x, v)
        Jv .= 0
        Jv[Omega1] .=  v[Omega1]
        Jv 
    end

    FirstOrderModel(obj, grad!, xinit, name = "MIT"),
    FirstOrderNLSModel(resid!, jprod_resid!, jprod_resid!, m*n, xinit, name = "MIT-LS") 
end

f, g = mat_rand_model(size(data, 1), size(data, 2), data[Omega], Om, vec(data)); 

# ROSolverOptions
options = ROSolverOptions(ν = 1., β = 1e16, ϵ = 1e-5, verbose = 10, maxIter = 1000);

# Terme non lisse
h = Rank(2500.,ones(943,1682));

sola = R2(f, h, options, x0 = f.meta.x0)
