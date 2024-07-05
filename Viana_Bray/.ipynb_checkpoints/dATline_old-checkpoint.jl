#Here I copy the code from dATline.c written by Maria Chiara

function oneStep(u::Array{Float64,2},δ::Array{Float64,2},T::Float64,Fields::Array{Float64,1}, Couplings::Array{Float64,2})
    
    """ 
    Function to perform one iteration of the cavity method at temperature T
    
    parameters:
    - u: messages u_i->j for all connected i,j (in form of a LxL matrix)
    - δ: perturbations to the fixed point state
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - new_u: new values of the messages after an iteration
    - new_δ: new values of the perturbations after an iteration
    """

    L = length(Fields)
    β = 1/T
    new_u = zeros(L,L)
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
            new_u[i,j] = T * atanh( tanh(β*Couplings[i,j]) * tanh(β*ContributionSum)) ## Eq(32) in M.C. Angelini notes
            
            ## Now for the perturbations I don't know around which values of u I should perturb the system. 
            ## I try two things below: perturb the u = 0 state or the u^* state (fixed point). Comment the one you don't want
            ## The equation is the derivative of Eq(32) w.r.t. u_k->j

            new_δ[i,j] = tanh(β*Couplings[i,j]) * (1 - tanh(β*ContributionSum)^2)/ (1 - (tanh(β*Couplings[i,j]) * tanh(β*ContributionSum))^2)*ContributionPerturbation #check if the found u^* state is stable
            #new_δ[i,j] = tanh(β*Couplings[i,j]) * (1 - tanh(β*Fields[i])^2)/ (1 - (tanh(β*Couplings[i,j]) * tanh(β*Fields[i]))^2)*ContributionPerturbation #check if u=0 is stable
        end
    end
    return new_u, new_δ
end

function ManySteps(Niter::Int,u::Array{Float64,2},δ::Array{Float64,2},T::Float64,Fields::Array{Float64,1}, Couplings::Array{Float64,2})

    """ 
    Function to perform multiple iterations of the cavity method at temperature T
    
    parameters:
    - Niter: number of iterations
    - u: initial messages u_i->j for all connected i,j (in form of a LxL matrix)
    - δ: initial perturbations to the fixed point state
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - u_new: new values of the messages after all the iterations
    - δ_new: new values of the perturbations after all the iterations
    - DifferenceEvolution: Frobenius norm of u^t+1-u^t for each step t
    - AveragePerturbationEvolution: Average of δ^t for each step t
    - MaximumPerturbationEvolution: Maximum of δ^t for each step t
    """

    DifferenceEvolution = zeros(Niter)
    AveragePerturbationEvolution = zeros(Niter)
    MaximumPerturbationEvolution = zeros(Niter)
    u_iter = copy(u)
    δ_iter = copy(δ)
    u_new = zeros(L,L)
    δ_new = zeros(L,L)
    for iter in 1:Niter
        u_new, δ_new = oneStep(u_iter,δ_iter,T,Fields,Couplings)
        DifferenceNorm = 0
        for i in 1:L    
            for j in 1:L
                DifferenceNorm += (u_new[i,j] - u_iter[i,j])^2
            end
        end
        AveragePerturbationEvolution[iter] = mean(δ_new)
        MaximumPerturbationEvolution[iter] = maximum(abs.(δ_new))
        DifferenceEvolution[iter] = sqrt(DifferenceNorm)
        u_iter = copy(u_new)
        δ_iter = copy(δ_new)
    end
    return u_new, δ_new, DifferenceEvolution, AveragePerturbationEvolution, MaximumPerturbationEvolution

end

function dATlineSearch(T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2}, Iterations::Int)

    """ 
    Function to perform the dATline analysis of a single realization of some model using the cavity method at temperature T
    
    parameters:
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - U: new values of the messages after all the iterations
    - Δ: new values of the perturbations after all the iterations
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
            δ[i,j] = 1.0
        end
    end

    U, Δ, ev, avPert, maxPert = ManySteps(Iterations,u,δ,T,Fields,Couplings)

    return U,Δ,ev,avPert,maxPert

end


function dATlineSearch(T::Float64,Fields::Array{Float64,1},Couplings::Array{Float64,2}, Iterations::Int, messages::Array{Float64,2}; pert::Float64 = 1.0)

    """ 
    Function to perform the dATline analysis of a single realization of some model using the cavity method at temperature T
    
    parameters:
    - T: temperature
    - Fields: random fields acting on each spin
    - Couplings: random couplings between spins

    output:
    - U: new values of the messages after all the iterations
    - Δ: new values of the perturbations after all the iterations
    - ev: Frobenius norm of u^t+1-u^t for each step t
    - avPert: Average of δ^t for each step t
    - maxPert: Maximum of δ^t for each step t
    """

    L = length(Fields)
    δ = zeros(L,L)

    for i in 1:L 
        for j in findall(x -> x != 0, Couplings[i,:])
            δ[i,j] = pert
        end
    end

    U, Δ, ev, avPert, maxPert = ManySteps(Iterations,messages,δ,T,Fields,Couplings)

    return U,Δ,ev,avPert,maxPert

end
