using Pkg

print(Threads.nthreads())

Pkg.activate(".")

using Distributions, LinearAlgebra, MiCRM, EasyFit, Distances, StatsBase, Random
using OrdinaryDiffEq, DiffEqCallbacks
using JLD2

#solve system with stressor
#derivatives
function growth_MiCRM_detox!(dx,x,p,t,i)
    #mortality
    dx[i] += -p.m[i] * x[i]
    #resource uptake
    for α = 1:p.M
        tmp = 0.0
        for β = 1:p.M
            tmp += p.l[α,β]
        end
        dx[i] += x[α + p.N] * x[i] * p.u[i,α] * (1 - tmp) * p.kw.γ[i,α]
    end
end

#derivative function
dx!(dx,x,p,t) = MiCRM.Simulations.dx!(dx,x,p,t;  growth! = growth_MiCRM_detox!)

#params
#Dirchlet U matricies
#Dirchlet U matricies
function dirchlet_uptake(N,M,kw)
    u = zeros(N,M)
    u[:,1 : end-1] .= Array(rand(Dirichlet(M-1, kw[:au]), N)')
    u[:,end] .= rand(Dirichlet(N, 1.0), 1)
    return(u)
end

#Dirchlet L matricies
function dirchlet_leakage(N,M,kw)
    l = zeros(M,M)
    # for α = 1:M-2
    #     l[α, α+1 : end-1] .= rand(Dirichlet(M - 1 - α, kw[:al]), 1)
    # end
    l[1 : end-1, 1 : end-1] .= Array(rand(Dirichlet(M-1, kw[:al]), M-1)')
    return(l * kw[:λ])
end

function fρ(N,M,kw)
    ρ = ones(M) * M
    # ρ[1] = M * M
    ρ[end] = 0.0
    return(ρ)
end

N,M = 25,25

#gamma
γ = ones(N,M)
γ[:,end] .= -1

#generate test parameters
p = MiCRM.Parameters.generate_params(N, M, f_u = dirchlet_uptake, f_l = dirchlet_leakage, f_ρ = fρ,
    λ = 1e-6, γ = γ, au = 1.0, al = 1.0)

#simualtion params
x0 = rand(N+M)
t = (0.0,1e10)

prob = ODEProblem(dx!, x0, t, p)
sol = solve(prob, AutoTsit5(Rosenbrock23()), callback = TerminateSteadyState())

#vary stressor supply, and uptake & leakage structure
N_r,N_u,N_λ,N_ρ = 100,20,3,5

u_vec = 10 .^ range(-2, 0, length = N_u)
λ_vec = [0.1, 0.3, 0.7]
ρ_vec = [0, 0.1, 1.0, M, 150]



sol_mat = Array{Any, 4}(undef, N_r,N_u,N_λ,N_ρ)
mass_mat = similar(sol_mat)
p_mat = similar(sol_mat)
J_mat = similar(sol_mat)
dx_mat = similar(sol_mat)

Threads.@threads for r = 1:N_r
    for (i,u) = enumerate(u_vec)
        for (j,λ) = enumerate(λ_vec)
             # println("u: ", i, " l: ", j)
             # println("Thread: ", Threads.threadid()," rep: ",r ," λ: ", λ)
             p_sim = MiCRM.Parameters.generate_params(N, M, f_u = dirchlet_uptake, f_l = dirchlet_leakage, f_ρ = fρ, λ = λ, γ = γ, au = u, al = 1.0)
            for (k,ρ) = enumerate(ρ_vec)
                # Random.seed!(ρ_ind)
                p_sim.ρ[end] = ρ

                prob = ODEProblem(dx!, x0, t, p_sim)
                sol = solve(prob, Rosenbrock23(),
                    callback = TerminateSteadyState(), save_everystep = false)

                println(sol.retcode, "  ", maximum(abs.(sol(sol.t[end], Val{1}))))
                  
                mass_mat[r,i,j,k] = deepcopy(sol[end])
                p_mat[r,i,j,k] = deepcopy(p_sim)
                J_mat[r,i,j,k] = MiCRM.Analysis.get_jac(sol)
                dx_mat[r,i,j,k] = (sol.retcode, maximum(abs.(sol(sol.t[end],Val{1}))))

            end 
        end
    end
end

println("saving")

save("./data/detox_simulations.jld2", Dict("mass" => mass_mat, "p" => p_mat, "J" => J_mat, "dx" => dx_mat))