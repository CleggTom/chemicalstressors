using Pkg

println(Threads.nthreads())

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

#get separate U and L matricies
dgdR(p, R, i, a) = p.u[i,a] * p.kw.γ[i,a] * (1 - p.kw.λ)

function dLdC(p, R, j, a)
    v = 0.0
    for b = 1:p.M
        v += p.l[b,a] * R[b] * p.u[j,b]
    end
    return v
end

dUdC(p, R, j, a) = R[a] * p.u[j,a]

function a_L(p,R,i,j)
    v = 0.0
    for a = 1:p.M
        v += dgdR(p,R,i,a) * dLdC(p,R,j,a)
    end
    return(v)
end

function a_U(p,R,i,j)
    v = 0.0
    for a = 1:p.M
        v += dgdR(p,R,i,a) * dUdC(p,R,j,a)
    end
    return(v)
end

get_U_mat(p,R) = [a_U(p,R,i,j) for i = 1:p.N , j = 1:p.N]
get_L_mat(p,R) = [a_L(p,R,i,j) for i = 1:p.N , j = 1:p.N]

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

a_U_mat = a_L_mat = similar(r["J"])
Rinf = similar(r["J"])
pert = similar(r["J"])

println(length(r["J"]))

prop = [0.0]

Threads.@threads for i = 1:length(r["J"])
    J = r["J"][i]
    p = r["p"][i]
    mass = r["mass"][i]
    
    prop[1] += 1 / length(r["J"])
    Threads.threadid() == 1 && println(prop[1])
    # a[i] = get_a_mat(r["J"][i], r["p"][i], r["mass"][i])
    
    if mass != nothing
        a_U_mat[i] = get_U_mat(p, mass[p.N+1 : end]) 
        a_L_mat[i] = get_L_mat(p, mass[p.N+1 : end])
        Rinf[i] = get_Rinf(J, p, mass);
        # @time pert[i] = max_perturbation(J, p, mass);
    end
end


save("./data/analysis.jld2", Dict("a_U" => a_U_mat, "a_L" => a_L_mat, "Rinf" => Rinf, "max_p" => pert))