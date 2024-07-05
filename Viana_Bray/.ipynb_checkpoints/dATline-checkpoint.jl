#Here I copy the code from dATline.c written by Maria Chiara

function oneStep_u(u::Array{Float64,2},T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})
    
    """ 
    Function to perform one iteration of the cavity method at temperature T
    
    parameters:
    - u: messages u_i->j for all connected i,j (in form of a LxL matrix)
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - new_u: new values of the messages after an iteration
    """

    L = length(Fields)
    β = 1/T
    new_u = zeros(L,L)

    for i in 1:L
        Neighbors = findall(x -> x != 0, Couplings[i,:]) ## all the spins in the neighborhood of i
        for j in Neighbors
            ContributionSum = Fields[i] ## contribution to the u_i->j message coming from the field
            for k in filter(x -> x != j, Neighbors)
                ContributionSum += u[k,i] ## contribution to the u_i->j message coming from the other messages
            end
            new_u[i,j] = T * atanh( tanh(β*Couplings[i,j]) * tanh(β*ContributionSum)) ## Eq(32) in M.C. Angelini notes
        end
    end
    return new_u
end

function oneStep_delta(u::Array{Float64,2},δ::Array{Float64,2},T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})
    
    """ 
    Function to perform one iteration of the cavity method at temperature T
    
    parameters:
    - u: messages u_i->j at the fixed point
    - δ: perturbations to the fixed point state
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - new_δ: new values of the perturbations after an iteration
    """

    L = length(Fields)
    β = 1/T
    new_δ = zeros(L,L)

    for i in 1:L
        Neighbors = findall(x -> x != 0, Couplings[i,:]) ## all the spins in the neighborhood of i
        for j in Neighbors
            ContributionSum = Fields[i] ## contribution to the u_i->j message coming from the field
            ContributionPerturbation = 0.0
            for k in filter(x -> x != j, Neighbors)
                ContributionSum += u[k,i] ## contribution to the u_i->j message coming from the other messages
                ContributionPerturbation += δ[k,i] ## contribution to the δ_i->j perturbation coming from the other perturbations
            end
            
            ## I perturb the u^* state (fixed point).
            ## The equation is the derivative of Eq(32) w.r.t. u_k->j
            new_δ[i,j] = tanh(β*Couplings[i,j]) * (1 - tanh(β*ContributionSum)^2)/ (1 - (tanh(β*Couplings[i,j]) * tanh(β*ContributionSum))^2)*ContributionPerturbation #check if the found u^* state is stable
        
        end
    end
    return new_δ
end

function ManySteps_u(Niter::Int,u::Array{Float64,2},T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})

    """ 
    Function to perform multiple iterations of the cavity method at temperature T
    
    parameters:
    - Niter: number of iterations
    - u: the starting values of u
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - u_new: new values of the messages after all the iterations
    - DifferenceEvolution: Frobenius norm of u^t+1-u^t for each step t
    """

    DifferenceEvolution = zeros(Niter)
    u_iter = copy(u)
    u_new = zeros(L,L)

    for iter in 1:Niter
        u_new = oneStep_u(u_iter,T,Fields,Couplings)
        DifferenceNorm = 0
        for i in 1:L    
            for j in 1:L
                DifferenceNorm += (u_new[i,j] - u_iter[i,j])^2
            end
        end
        DifferenceEvolution[iter] = sqrt(DifferenceNorm)
        u_iter = copy(u_new)
    end
    return u_new, DifferenceEvolution
end


function ManySteps_delta(Niter::Int,u::Array{Float64,2},δ::Array{Float64,2},T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})

    """ 
    Function to perform multiple iterations of the cavity method at temperature T for the perturbations
    
    parameters:
    - Niter: number of iterations
    - u: the fixed point values of u
    - δ: the starting value of the perturbations
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - δ_new: new values of the perturbations after all the iterations
    - AveragePerturbationEvolution: Average of δ^t for each step t
    - MaximumPerturbationEvolution: Maximum of δ^t for each step t
    """

    AveragePerturbationEvolution = zeros(Niter)
    MaximumPerturbationEvolution = zeros(Niter)
    δ_iter = copy(δ)
    δ_new = zeros(L,L)

    for iter in 1:Niter
        δ_new = oneStep_delta(u,δ_iter,T,Fields,Couplings)
        AveragePerturbationEvolution[iter] = mean(δ_new)
        MaximumPerturbationEvolution[iter] = maximum(abs.(δ_new))
        δ_iter = copy(δ_new)
    end
    return δ_new, AveragePerturbationEvolution, MaximumPerturbationEvolution

end


function dATlineSearch(Niter_u::Int,Niter_delta::Int,δstart::Float64,T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})

    """ 
    Function to perform the dATline analysis of a single realization of some model using the cavity method at temperature T
    
    parameters:
    - Niter_u: number of iterations to find u^*
    - Niter_delta: number of iterations to find delta
    - δstart: starting value of all deltas (uniform)
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - U: new values of the messages after all the Niter_u iterations
    - Δ: new values of the perturbations after all the Niter_delta iterations
    - ev: Frobenius norm of u^t+1-u^t for each step t
    - avPert: Average of δ^t for each step t
    - maxPert: Maximum of δ^t for each step t
    """

    L = length(Fields)
    u = zeros(L,L)
    δ = zeros(L,L)

    for i in 1:L 
        for j in findall(x -> x != 0, Couplings[i,:])
            u[i,j] = -1 .+ 2 .* rand()
            δ[i,j] = δstart
        end
    end

    U, ev = ManySteps_u(Niter_u,u,T+0.2,Fields,Couplings)
    
    Δ, avPert, maxPert = ManySteps_delta(Niter_delta,U,δ,T,Fields,Couplings)

    return U,Δ,ev,avPert,maxPert

end


function stability_analysis(Niter_u::Int,T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2})
    
    L = length(Fields)
    u = zeros(L,L)
    β = 1. / T

    for i in 1:L 
        for j in findall(x -> x != 0, Couplings[i,:])
            u[i,j] = -1 .+ 2 .* rand()
        end
    end

    U, ev = ManySteps_u(Niter_u,u,T,Fields,Couplings)
    
    matr = zeros(L,L)
    for i in 1:L
        Neighbors = findall(x -> x != 0, Couplings[i,:]) ## all the spins in the neighborhood of i
        for j in Neighbors
            ContributionSum = Fields[i] ## contribution to the u_i->j message coming from the field
            ContributionPerturbation = 0.0
            for k in filter(x -> x != j, Neighbors)
                ContributionSum += U[k,i] ## contribution to the u_i->j message coming from the other messages
            end
            
            ## I perturb the u^* state (fixed point).
            ## The equation is the derivative of Eq(32) w.r.t. u_k->j
            matr[i,j] = tanh(β*Couplings[i,j]) * (1 - tanh(β*ContributionSum)^2)/ (1 - (tanh(β*Couplings[i,j]) * tanh(β*ContributionSum))^2) #check if the found u^* state is stable
        
        end
    end 
    
    return matr
end


