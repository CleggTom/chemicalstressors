using Pkg

print(Threads.nthreads())

Pkg.activate(".")

using Distributions, LinearAlgebra, MiCRM, EasyFit, Distances, StatsBase
using OrdinaryDiffEq
using JLD2
using Optim

#Interactions
#get interaction matrix based on and J the solution
function get_a_mat(J, p, mass)
    #remove extinct consumers
    C_extant = findall(mass[1:p.N] .> eps())
    N_new = length(C_extant)
    indx = vcat(C_extant, (p.N+1) : (p.N + p.M))

    #new jacobian
    J_ = J[indx,indx]

    # @assert (eigvals(J_)[end].re) < 0

    a_mat = zeros(N_new,N_new)
    for j = 1:N_new
        store = J_[j,:]
        J_[j,:] .= 0.0
        a_mat[j,:] = Real.(eigvecs(J_)[:,findall(eigvals(J_) .== 0.0)])[1:N_new]
        
        a_mat[j,:] .= a_mat[j,:] ./ a_mat[j,j]

        J_[j,:] .= store
    end
    return(a_mat)
end

function get_Rinf(J, p, mass)
    #remove extinct consumers
    C_extant = findall(mass[1:p.N] .> eps())
    N_new = length(C_extant)
    indx = vcat(C_extant, (p.N+1) : (p.N+p.M))
    #new jacobian
    J_ = J[indx,indx]
    
    return(eigvals(J[indx,indx])[end])
end

function max_perturbation(J,p,mass)

    # #remove extinct consumers
    C_extant = findall(mass[1:p.N] .> eps())
    N_new = length(C_extant)
    indx = vcat(C_extant, (p.N+1) : (p.N + p.M))

    # #new jacobian
    J_ = J[indx,indx]

    u = zeros(length(indx))
    u[end] = 0.1

    w = zeros(length(indx))
    w[1:N_new] .= 1.0


    fx(t) = -norm(w .* exp(J_ * t[1]) * u)
    
    lower = [eps()]
    upper = [Inf]
    initial_x = [0.03]
    inner_optimizer = GradientDescent()
    
    r1 = optimize(fx, lower, upper, initial_x, Fminbox(inner_optimizer))
    
    return(r1.minimum, r1.minimizer)
end

#testing
function mean_off_diag(A)
    (sum(A) - sum(A[diagind(A)])) / prod(size(A) .- [0,1])
end

r = JLD2.load("./data/detox_simulations.jld2")

a = similar(r["J"])
Rinf = similar(r["J"])
pert = similar(r["J"])

prop = [0.0]

Threads.@threads for i = 1:length(r["J"])
    
    prop[1] += 1 / length(r["J"])
    Threads.threadid() == 1 && println(prop[1])
    a[i] = get_a_mat(r["J"][i], r["p"][i], r["mass"][i])
    Rinf[i] = get_Rinf(r["J"][i], r["p"][i], r["mass"][i]);
    pert[i] = max_perturbation(r["J"][i], r["p"][i], r["mass"][i]);
end


save("./data/analysis.jld2", Dict("a_mat" => a, "Rinf" => Rinf, "max_p" => pert))