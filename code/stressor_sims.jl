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
function fu(N,M,kw)
    u = zeros(N,M)
    u[:, 1 : (M-1)] .= MiCRM.Parameters.modular_uptake(N,M - 1, N_modules = 5, s_ratio = kw[:u_ratio]) .* kw[:u_tot]
    u[:, end] .= ones(N)
    return(u)
end

function fl(N,M,kw)
    l = MiCRM.Parameters.modular_leakage(M; N_modules = 5, s_ratio = kw[:l_ratio], λ = kw[:λ])
    #dont leak to or from stressor
    l[:,end] .= 0.0
    l[end,:] .= 0.0
    [l[i,:] .=  kw[:λ] * l[i,:] ./ sum(l[i,:]) for i = 1:(M-1)]
    return(l)
end

fρ(N,M,kw) = ones(M) * M 

N,M = 25,25

#gamma
γ = ones(N,M)
γ[:,end] .= -1

#generate test parameters
p = MiCRM.Parameters.generate_params(N, M, f_u = fu, f_l = fl, f_ρ = fρ, λ = 1e-6, γ = γ, u_ratio = 10.0, l_ratio = 10.0, u_tot = 1.0)

#simualtion params
x0 = rand(N+M)
t = (0.0,1e10)

prob = ODEProblem(dx!, x0, t, p)
sol = solve(prob, AutoTsit5(Rosenbrock23()), callback = TerminateSteadyState())


#vary stressor supply, and uptake & leakage structure
N_r,N_u,N_l,N_utot,N_λ,N_ρ = 10,10,10,5,3,5

s_vec = 10 .^ range(0, 2, length = N_u)
u_vec = range(1.0, 10.0, length = N_utot)
λ_vec = [0.1, 0.3, 0.7]
ρ_vec = [0.0, 0.1, 1.0,10.0, M]



sol_mat = Array{Any, 6 }(undef, N_r,N_u,N_l,N_utot,N_λ,N_ρ)
mass_mat = similar(sol_mat)
p_mat = similar(sol_mat)
J_mat = similar(sol_mat)

Threads.@threads for r = 1:N_r
    # println(r)
    for (i,us) = enumerate(s_vec)
        for (j,ls) = enumerate(s_vec)
            for (k, utot) = enumerate(u_vec)
                for (l,λ) = enumerate(λ_vec)
                

                    println("u: ", i, " l: ", j)
                    println("Thread: ", Threads.threadid()," rep: ",r ," λ: ", λ)

                    p_sim = MiCRM.Parameters.generate_params(N, M, f_u = fu, f_l = fl, f_ρ = fρ, λ = λ, γ = γ, u_ratio = us, l_ratio = ls, u_tot = utot)

                    for (ρ_ind,ρ) = enumerate(ρ_vec)

                        # Random.seed!(ρ_ind)
                        p_sim.ρ[end] = ρ

                        prob = ODEProblem(dx!, x0, t, p_sim)
                        sol = solve(prob, Rosenbrock23(),
                            callback = TerminateSteadyState(), save_everystep = false)

                        if sol.retcode == ReturnCode.Terminated
                            mass_mat[r,i,j,k,l,ρ_ind] = deepcopy(sol[end])
                            p_mat[r,i,j,k,l,ρ_ind] = deepcopy(p_sim)
                            J_mat[r,i,j,k,l,ρ_ind] = MiCRM.Analysis.get_jac(sol, thresh = 1.0)
                        else
                            mass_mat[r,i,j,k,l,ρ_ind] = p_mat[r,i,j,k,l,ρ_ind] = J_mat[r,i,j,k,l,ρ_ind] = nothing
                        end

                    end
                end
            end 
        end
    end
end

println("saving")

save("./data/detox_simulations.jld2", Dict("mass" => mass_mat, "p" => p_mat, "J" => J_mat))