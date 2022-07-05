include("Shifted_Rank_f.jl")

using Test
## ROSOLVERS
options = ROSolverOptions(ν = 1., β = 1e16, ϵ = 1e-6, verbose = 10)

# Simple prox
n = 10
λ = 10.
st1 = rand(n)
x = vec(reshape(Diagonal(st1),n^2,1))
q = x.^2
s = x/2
F = psvd_workspace_dd(zeros(n,n), full=false)
h = Rank(λ,ones(n,n),F);
y = zeros(n^2);

f = lrcomp_model(10,10);


Sol1 = PGa(f[1], h, options, x0=f.meta.x0)
Sol2 = R2a(f[1], h, options, x0=f.meta.x0)

prox!(y, h, vec(reshape(f[2], 100,1)), 1.)

@test all(y .≈ Sol1.solution)
@test all(y .≈ Sol2.solution)



## Random completion
n1=100;
f2 = mat_rand_model(100,100,30,0.8,sqrt(0.0001),sqrt(0.1), 0.2);
Fr = psvd_workspace_dd(zeros(n1,n1), full=false);
hr = Rank(20.,ones(n1,n1),Fr);

Solr1 = PGa(f2[1], hr, options, x0=f2[1].meta.x0)
Solr2 = R2a(f2[1], hr, options, x0=f2[1].meta.x0)
