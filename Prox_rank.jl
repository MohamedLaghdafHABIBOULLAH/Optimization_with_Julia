using LinearAlgebra
using ProximalOperators


function prox_rank(f::NormL0, gamma, A)
    F = svd(A)
    return F.U * Diagonal(ProximalOperators.prox_naive(f,F.S,gamma)[1]) * F.Vt
end

