function H_distance(seq1::Vector, seq2::Vector)
    ### Function computing the Hamming distance between two vectors

    L = length(seq1)
    @assert L == length(seq2) "Error: the two vectors do not have the same length"
    d = 0
    for i in 1:L
        if seq1[i] != seq2[i]
            d += 1
        end
    end
    return d
end

function H_distance(seq1::Matrix, seq2::Matrix)
    ### Function computing the Hamming distance between two matrices

    L, _ = size(seq1)
    @assert L == size(seq2)[1] "Error: the two vectors do not have the same length"
    d = 0
    for i in 1:L
        for j in 1:L
            if seq1[i,j] != seq2[i,j]
                d += 1
            end
        end
    end
    return d
end


########################################
####### GIBBS SAMPLING TOOLS ###########
########################################

function cond_proba(seq::Array{Int,1}, h::Float64, J::Array{Float64,1}, L::Int, β::Float64)

    """
    Function to compute the conditional probability used in the Gibbs sampling procedure

    parameters:
    - seq: the configuration under study
    - h: the local fields acting on the site under study 
    - J: the couplings acting on the site under study
    - L: the size of the system
    - β: inverse temperature used to compute the probability

    output:
    - prob: vector of 2 elements storing the probability of having -1 and +1 in the position under study
    """

	prob = Array{Float64, 1}(undef, 2)
    @inbounds for q_k in 1:2
		log_proba = h

        for j in 1:L
            log_proba += J[j]*seq[j] 
        end

        log_proba *= ((q_k-1)*2-1)

		prob[q_k] = exp(β*log_proba)   
	end
	return normalize(prob,1)
end


function MCMC_Gibbs_onestep(seq::Vector{Int}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform one MonteCarlo step with the Gibbs setting.

    parameters:
    - seq: the initial configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - seq: new sequence after one step of sampling
    """ 

    new_seq = copy(seq)
    mut_site = rand(1:L)    # choose a random site
    mut_amino = sample([-1,1], ProbabilityWeights(cond_proba(seq,h[mut_site],J[:,mut_site],L,β)))   # choose a random value for the site following the conditional distribution
    new_seq[mut_site] = mut_amino
    return new_seq
end

#############################################
####### METROPOLIS SAMPLING TOOLS ###########
#############################################


function Delta_energy(idx::Int, h::Array{Float64,1}, J::Array{Float64,2}, seq::Array{Int,1})

    """
    Function to compute the variation in energy between two configurations. Valid for only one spin change.

    parameters:
    - idx: position of the mutated spin
    - h: the local fields  
    - J: the couplings 
    - seq: the configuration

    output:
    - ΔE: variation in energy when flipping spin idx
    """

    L = length(h)
    _ΔE = 0.0
    
    _ΔE += h[idx]
    for j in findall(x -> x != 0, J[idx,:])
        _ΔE += J[idx,j]*seq[j] 
    end
    ΔE = 2*seq[idx]*_ΔE
    
    return ΔE
end


function MCMC_Metropolis_onestep(seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform one MonteCarlo step with the Metropolis setting.

    parameters:
    - seq: the starting configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - seq: new sequence after one step of sampling
    - ΔE: energy variation after the step
    """ 

    mut = rand(1:L)    # choose a random site

    ΔE = Delta_energy(mut,h,J,seq)

    if ΔE < 0
        seq[mut] *= -1
        return seq, ΔE
    elseif rand() < exp(-β*ΔE)
        seq[mut] *= -1
        return seq, ΔE
    end
    return seq, 0.0
end

function MCMC_Metropolis_onestepnoene(seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform one MonteCarlo step with the Metropolis setting (does not give back the energy variation).

    parameters:
    - seq: the starting configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - seq: new sequence after one step of sampling
    """ 

    mut = rand(1:L)    # choose a random site 

    ΔE = Delta_energy(mut,h,J,seq)

    if ΔE < 0
        seq[mut] *= -1
    elseif rand() < exp(-β*ΔE)
        seq[mut] *= -1
    end
    return seq
end

function MCMC_Metropolis(nsteps::Int, seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform a MonteCarlo trajectory with the Metropolis setting.

    parameters:
    - nsteps: the number of steps to be performed
    - seq: the starting configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - seq: final configuration
    """ 

    _seq = deepcopy(seq)

    for i in 1:nsteps
        _seq, _ = MCMC_Metropolis_onestep(_seq, L, h, J, β)
    end
    return _seq
end

function MCMC_Metropolis_store_en_magn(nstore::Int, nsteps::Int, seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform a MonteCarlo trajectory with the Metropolis setting. This function stores the energy of the configuration, its magnetization, 
    and the overlap with the initial configuration once every nsteps/nstore steps
    ATTENTION: my definition of overlap, I should check which definition I should use

    parameters:
    - nstore: the number of stored values of energy, magnetization, and overlap
    - nsteps: the number of steps to be performed
    - seq: the initial coniguration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - en: vector of length nstore containing the energy of the configuration at each writing step
    - magn: vector of length nstore containing the magnetization of the configuration at each writing step
    - overlap: vector of length nstore containing the overlap of the configuration with the initial one at each writing step
    - seq: final configuration
    """ 

    @assert nsteps%nstore == 0 "Please choose a value of nstore that allows for an integer number of steps between storing"
    steps_storing = nsteps÷nstore
    en = Array{Float64}(undef, nstore)
    magn = Array{Float64}(undef, nstore)
    overlap = Array{Float64}(undef, nstore)
    en_ev = compute_energy(h, J, seq) 
    _seq = deepcopy(seq)
    for i in 1:nsteps
        _seq, ΔE = MCMC_Metropolis_onestep(_seq, L, h, J, β)
        en_ev += ΔE
        if i%steps_storing == 0  
            en[i÷steps_storing] = en_ev
            magn[i÷steps_storing] =  sum(_seq)/L
            overlap[i÷steps_storing] = (L-2*H_distance(_seq,seq))/L
        end
    end
    return en,magn,overlap,_seq
end


function Multiple_MCMC_Metropolis(nsamples::Int, nsteps::Int, seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform many MonteCarlo trajectories with the Metropolis setting.

    parameters:
    - nsamples: number of samples to be computed
    - nsteps: the number of steps to be performed
    - seq: the initial configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - MSA: MSA of configurations after nsteps
    """ 

    evolving_MSA = zeros(Int64,(nsamples,L))
    for i in 1:nsamples
        evolving_MSA[i,:] = copy(seq)
    end

    en = zeros(nsteps)
    di = zeros(nsteps)

    for s in 1:nsamples
        evolving_MSA[s,:] .= MCMC_Metropolis(nsteps,evolving_MSA[s,:], L, h, J, β)
    end

    return evolving_MSA
end

function Multiple_MCMC_Metropolis(nsamples::Int, nsteps::Int, n_store::Int, seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)

    """
    Function to perform many MonteCarlo trajectories with the Metropolis setting, saving sum 

    parameters:
    - nsamples: number of samples to be computed
    - nsteps: the number of steps to be performed
    - seq: the initial configuration
    - L: the size of the system
    - h: the local fields  
    - J: the couplings 
    - β: inverse temperature used to compute the probability

    output:
    - MSA: MSA of configurations of some steps
    """ 

    evolving_MSA = zeros(Int64,(nsamples,L))
    for i in 1:nsamples
        evolving_MSA[i,:] = copy(seq)
    end

    en = zeros(nsteps)
    di = zeros(nsteps)

    for s in 1:nsamples
        evolving_MSA[s,:] .= MCMC_Metropolis(nsteps,evolving_MSA[s,:], L, h, J, β)
    end

    return evolving_MSA
end

#################
# PROTEINS CODE #
#################


# #############################################
# ####### ENTROPY COMPUTATION #################
# #############################################

# function Multiple_MCMC_Metropolis_entropyevolution(nsamples::Int, nstore::Int, nsteps::Int, seq::Array{Int,1}, L::Int, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)
#     """
#     Function to perform many MonteCarlo trajectories with the Metropolis setting. This function computes the entropy of each site.

#     parameters:
#     - nsamples: number of samples to be computed
#     - nstore: the number of stored values of energy and distance
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: vector of size Length_of_sequences x nstore containing the entropy of the sequence at each writing step for each site

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)
#     evolving_MSA = zeros(Int64,(L,nsamples))
#     for i in 1:nsamples
#         evolving_MSA[:,i] = copy(seq)
#     end

#     for n in 1:nsteps
#         for i in 1:nsamples
#             evolving_MSA[:,i] .= MCMC_Metropolis_onestepnoene(evolving_MSA[:,i], L, h, J, β)
#         end
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myentropy(evolving_MSA,2)
#         end
#         if n%(nsteps/100) == 0
#             println("we are at step $n")
#         end
#     end

#     return entr_vec
# end

 function compute_empirical_freqs(Z::Array{Int64,2}, q::Int) 
     """
     Function to compute the frequency of occurrence of each spin in the MSA.

     parameters:
     - Z: MSA in format Number_of_sequences x Length_of_sequences x Length_of_sequences 
     - q: length of the alphabet (2)

     output:
     - f: matrix q x Length_of_sequences with the frequence of each aminoacid at each site

     """
     M, L = size(Z)
     f = zeros(q, L)
     for i in 1:L
         for s in 1:M
             f[Int((Z[s, i]+1)/2+1), i] += 1
         end
     end
     return f ./= M
 end

function myentropy(Z::Array{Int64,2}, q::Int)
     """
     Function to compute the entropy of each site from the MSA.

     parameters:
     - Z: MSA in format Length_of_sequences x Number_of_sequences
     - q: length of the alphabet
     output:
     - entr: vector of length Length_of_sequences with the entropy of each site

     """
     M, L = size(Z)
     entr = zeros(L)
     f = compute_empirical_freqs(Z,q)
     @inbounds for i in 1:L
         _s = 0.0
         for a in 1:q
             # _s -= f[a, i] > 0 ? f[a, i] * log(f[a, i]) : 0.0  # check the base of the logarithm
             _s -= f[a, i] > 0 ? f[a, i] * log2(f[a, i]) : 0.0
         end
         entr[i] = _s
     end
     return entr
end


# function myCDE(context_seq::Array{Int,1}, h::Array{Float64,1}, J::Array{Float64,2}, β::Float64)
#     """
#     Function to compute the context-dependent entropy of each site from the model.

#     parameters:
#     - context_seq: sequence giving the context in which we compute the entropy
#     - h: the local fields of the model used to compute the entropy
#     - J: the couplings of the model used to compute the entropy
#     - β: inverse temperature used to compute the probability

#     output:
#     - entr: vector of length Length_of_sequences with the context-dependent entropy of each site

#     """
#     L = length(context_seq)
#     entr = zeros(L)
#     @inbounds for i in 1:L
#         _s = 0.0
#         f = cond_proba(context_seq,h[i],J[:,i],L,β)
#         for a in 1:2
#             # _s -= f[a] > 0 ? f[a] * log(f[a]) : 0.0  # check the base of the logarithm
#             _s -= f[a] > 0 ? f[a] * log2(f[a]) : 0.0
#         end
#         entr[i] = _s 
#     end
#     return entr
# end

# function myCDEsingle(site::Int, context_seq::Array{Int,1}, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to compute the context-dependent entropy of a single site from the inferred model.
#     ATTENTION: differently from the CIE, this entropy is computed from the model and not from the data.

#     parameters:
#     - site: site under study
#     - context_seq: sequence giving the context in which we compute the entropy
#     - h: the local fields of the model used to compute the entropy
#     - J: the couplings of the model used to compute the entropy
#     - q: length of the alphabet
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr: context-dependent entropy of site

#     """
#     L = length(context_seq)
#     _s = 0.0
#     f = cond_proba(context_seq,h[:,site],J[:,:,:,site],q,L)
#     for a in 1:q
#         # _s -= f[a] > 0 ? f[a] * log(f[a]) : 0.0  # check the base of the logarithm
#         _s -= f[a] > 0 ? f[a] * log2(f[a]) : 0.0
#     end
#     return _s
# end

# function Multiple_MCMC_Metropolis_entropyevolution_siteconstrained(available_sites::Vector{Int}, nsamples::Int, nstore::Int, nsteps::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform many MonteCarlo trajectories with the Metropolis setting with only some available_sites to move.
#     This function computes the entropy of each site.

#     parameters:
#     - available_sites: the sites that can be mutated
#     - nsamples: number of samples to be computed
#     - nstore: the number of stored values of energy and distance
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: vector of size Length_of_sequences x nstore containing the entropy of the sequence at each writing step for each site

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)
#     evolving_MSA = zeros(Int64,(L,nsamples))
#     for i in 1:nsamples
#         evolving_MSA[:,i] = copy(seq)
#     end

#     for n in 1:nsteps
#         for i in 1:nsamples
#             evolving_MSA[:,i], _ = MCMC_Metropolis_onestep_siteconstrained(evolving_MSA[:,i], available_sites, L, h, J; β=β)
#         end
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myentropy(evolving_MSA,q)
#         end
#         if n%(nsteps/100) == 0
#             println("we are at step $n")
#         end
#     end

#     return entr_vec, evolving_MSA
# end

# function MCMC_Metropolis_CDEevolution(site::Int, nstore::Int, nsteps::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform a MonteCarlo trajectory with the Metropolis setting. This function computes the CDE of the site under study every nsteps÷nstore
#     steps. It also returns a vector containing the steps at which the site mutated and a vector containing the CDE when the site mutated.

#     parameters:
#     - site: the site under study
#     - nstore: the number of stored values of CDE
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size L x nstore containing the CDE of every site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)        
#     changes = []
#     entr_at_changes = []
#     for n in 1:nsteps
#         new_seq, _ = MCMC_Metropolis_onestep(seq, L, h, J; β=β)
#         if new_seq[site] != seq[site]
#             push!(changes,n)
#             push!(entr_at_changes,myCDE(seq,h,J,q)[site])
#         end
#         seq = copy(new_seq)
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myCDE(seq,h,J,q)
#         end
#     end

#     return entr_vec,changes,entr_at_changes
# end

# function MCMC_Metropolis_CDEandDistEvolution(site::Int, nstore::Int, nsteps::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform a MonteCarlo trajectory with the Metropolis setting. This function computes the CDE of the site under study every nsteps÷nstore
#     steps. It also returns a vector containing the steps at which the site mutated and a vector containing the CDE when the site mutated.

#     parameters:
#     - site: the site under study
#     - nstore: the number of stored values of CDE
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size L x nstore containing the CDE of every site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)    
#     dist_vec = Array{Int,1}(undef, nstore)    
#     changes = []
#     entr_at_changes = []
#     starting_seq = copy(seq)
#     for n in 1:nsteps
#         new_seq, _ = MCMC_Metropolis_onestep(seq, L, h, J; β=β)
#         if new_seq[site] != seq[site]
#             push!(changes,n)
#             push!(entr_at_changes,myCDE(seq,h,J,q)[site])
#         end
#         seq = copy(new_seq)
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myCDE(seq,h,J,q)
#             dist_vec[n÷steps_storing] = H_distance(starting_seq,seq)
#         end
#     end

#     return dist_vec, entr_vec,changes,entr_at_changes
# end

# function MCMC_Metropolis_CDEevolution_siteconstrained(site::Int, available_sites::Vector{Int}, nstore::Int, nsteps::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform a MonteCarlo trajectory with the Metropolis setting with only some available_sites to move. 
#     This function computes the CDE of the site under study every nsteps÷nstore
#     steps. It also returns a vector containing the steps at which the site mutated and a vector containing the CDE when the site mutated.

#     parameters:
#     - site: the site under study
#     - available_sites: the sites that can be mutated
#     - nstore: the number of stored values of CDE
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size L x nstore containing the CDE of every site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)    
#     changes = []
#     entr_at_changes = []
#     for n in 1:nsteps
#         new_seq, _ = MCMC_Metropolis_onestep_siteconstrained(seq, available_sites, L, h, J; β=β)
#         if new_seq[site] != seq[site]
#             push!(changes,n)
#             push!(entr_at_changes,myCDE(seq,h,J,q)[site])
#         end
#         seq = copy(new_seq)
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myCDE(seq,h,J,q)
#         end
#     end

#     return entr_vec,changes,entr_at_changes
# end

# function MCMC_Gibbs_CDEevolution(site::Int, nstore::Int, nsteps::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform many MonteCarlo trajectory with the Gibbs setting. This function computes the CDE of the site under study every nsteps÷nstore
#     steps. It also returns a vector containing the steps at which the site mutated and a vector containing the CDE when the site mutated.

#     parameters:
#     - site: the site under study
#     - nstore: the number of stored values of CDE
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size L x nstore containing the CDE of every site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     steps_storing = nsteps÷nstore
#     entr_vec = Array{Float64,2}(undef, L, nstore)    
#     changes = []
#     entr_at_changes = []
#     for n in 1:nsteps
#         new_seq = MCMC_Gibbs_onestep(seq, L, h, J, q; β=β)
#         if new_seq[site] != seq[site]
#             push!(changes,n)
#             push!(entr_at_changes,myCDE(seq,h,J,q)[site])
#         end
#         seq = copy(new_seq)
#         if n%steps_storing == 0  
#             entr_vec[:,n÷steps_storing] = myCDE(seq,h,J,q)
#         end
#     end

#     return entr_vec,changes,entr_at_changes
# end

# function MCMC_Metropolis_CDEmutation(nstore::Int, start_mutation::Int, nmutations::Int, seq::Vector{Int}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform many MonteCarlo trajectory with the Metropolis setting. This function computes the CDE of mutations every nmutations÷nstore. 

#     parameters:
#     - nstore: the number of stored values of mutations
#     - start_mutation: the number of mutations before starting collecting values
#     - nmutations: the number of mutations to be studied
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size 2 x nstore containing the CDE of the site that mutated and the site itself 

#     """ 
#     steps_storing = (nmutations-start_mutation)÷nstore
#     entr_vec = Array{Float64,2}(undef, 2, nstore)    
#     mut = 0
#     while mut < nmutations
#         new_seq, mut_happened = MCMC_Metropolis_onestep(seq, L, h, J; β=β)
#         if mut_happened == 1
#             mut += 1
#             site = [i for i in 1:L if new_seq[i] != seq[i]][1]
#             if mut>start_mutation && (mut-start_mutation)%steps_storing == 0
#                 entr_vec[1,(mut-start_mutation)÷steps_storing] = myCDE(seq,h,J,q)[site]
#                 entr_vec[2,(mut-start_mutation)÷steps_storing] = site
#             end
#         end
#         seq = copy(new_seq)
#     end

#     return entr_vec
# end

# function MCMC_Metropolis_CDEcorrelationtable(site::Int, nsamples::Int, nstore::Int, nsteps::Int, Z::Array{Int,2}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to perform many MonteCarlo trajectories with the Metropolis setting. This function computes the CDE of site and correlates it with the context
#     in which the site is. It also outputs configurations with large or small CDE for the given site 

#     parameters:
#     - site: the site under study
#     - nstore: the number of stored values of CDE
#     - nsteps: the number of steps to be performed
#     - seq: the modified sequence
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: matrix of size L x nstore containing the CDE of every site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     steps_storing = nsteps÷nstore
#     table = zeros(Float64,L,q)    
#     counting_table = zeros(Int,L,q)    
#     large_CDE_seqs = Dict{}()
#     small_CDE_seqs = Dict{}()
#     large_CDE_count = 1
#     small_CDE_count = 1
#     # start_seq = copy(seq)
#     for s in 1:nsamples
#         chosen_seq = rand(collect(1:length(Z[1,:])))
#         seq = copy(Z[:,chosen_seq])
#         for n in 1:nsteps
#             new_seq, _ = MCMC_Metropolis_onestep(seq, L, h, J; β=β)
#             seq = copy(new_seq)
#             if n%steps_storing == 0  
#                 CDE = myCDE(seq,h,J,q)
#                 for i in 1:L
#                     if i != site
#                         table[i,seq[i]] += CDE[site]
#                         counting_table[i,seq[i]] += 1
#                     end
#                 end

#                 if CDE[site] > 3
#                     large_CDE_seqs[large_CDE_count] = (CDE[site],seq)
#                     large_CDE_count += 1
#                 elseif CDE[site] < 1 
#                     small_CDE_seqs[small_CDE_count] = (CDE[site],seq)
#                     small_CDE_count += 1
#                 end
#             end
#         end
#     end
#     table ./= counting_table

#     return table, large_CDE_seqs, small_CDE_seqs
# end

# function CDEevolution_analysis(site::Int, evMSA::Array{Int, 2}, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to study the evolution contained in a MSA. This function computes the CDE of the site under study every nsteps÷nstore
#     steps. It also returns a vector containing the steps at which the site mutated and a vector containing the CDE when the site mutated.

#     parameters:
#     - site: the site under study
#     - evMSA: MSA containing the evolution
#     - h: the local fields  
#     - J: the couplings 
#     - q: length of the alphabet
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - entr_vec: vector of size nstore containing the CDE of site at each writing step 
#     - changes: vector containing the steps at which the site changed value
#     - entr_at_changes: vector containing the CDE when the site changed value

#     """ 
#     nsteps = length(evMSA[1,:])
#     println(nsteps)
#     # nsteps = 10000
#     entr_vec = Array{Float64,1}(undef, nsteps)    
#     # entr_vec = Array{Float64,2}(undef, L,nsteps)    
#     changes = []
#     entr_at_changes = []
#     seq = copy(evMSA[:,1])
#     entr_vec[1] = myCDEsingle(site,seq,h,J,q)
#     for n in 2:nsteps
#         new_seq = copy(evMSA[:,n])
#         CDE = myCDEsingle(site,seq,h,J,q)
#         if new_seq[site] != seq[site]
#             push!(changes,n)
#             push!(entr_at_changes,CDE)
#         end
#         seq = copy(new_seq)
#         # entr_vec[:,n] = CDE
#         entr_vec[n] = CDE
#         if n%10000 == 0
#             println("we are at $(n/nsteps*100)%")
#         end
#     end

#     return entr_vec,changes,entr_at_changes
# end

# function CDEmutation_analysis(evMSA::Array{Int, 2}, L::Int, h::Array{Float64,2}, J::Array{Float64,4}, q::Int; β::Float64 = 1.0)
#     """
#     Function to study the evolution contained in a MSA. This function computes the CDE of mutations every nmutations÷nstore. 

#     parameters:
#     - evMSA: MSA containing the evolution
#     - L: the length of the sequence
#     - h: the local fields  
#     - J: the couplings 
#     - q: length of the alphabet
#     - β(optional): inverse temperature used to compute the probability

#     output:
#     - en_vec: matrix of size nsamples x nstore containing the energy of the sequence at each writing step for each sample
#     - dist_vec: matrix of size nsamples x nstore containing the distance of the sequence from the starting one at each writing step for each sample

#     """ 
#     start_mutation = 5000
#     entr_vec = []    
#     mut = 0
#     seq = copy(evMSA[:,1])
#     nsteps = 50000
#     for n in 2:nsteps
#         new_seq = copy(evMSA[:,n])
#         if new_seq != seq
#             mut += 1
#             site = [i for i in 1:L if new_seq[i] != seq[i]][1]
#             if mut>start_mutation
#                 push!(entr_vec,(myCDE(seq,h,J,q)[site],site))
#             end
#         end
#         seq = copy(new_seq)
#     end

#     return entr_vec
# end



# #############################################
# ################# DMS UTILS #################
# #############################################

# function CDEDMS(site::Int,seq::Vector{Int},h::Array{Float64,2}, J::Array{Float64,4}, q::Int)
    
#     L = length(seq)
#     cdedms = Array{Float64,2}(undef,L,q)
#     dms = Array{Float64,2}(undef,L,q)
#     for i in 1:L
#         if i != site
#             for a in 1:q
#                 if seq[i] == a
#                     cdedms[i,a] = NaN
#                     dms[i,a] = NaN
#                 else
#                     try_seq = copy(seq)
#                     try_seq[i] = a
#                     cdedms[i,a] = myCDEsingle(site,try_seq,h,J,q)
#                     dms[i,a] = compute_energy_single_sequence(h, J, try_seq)
#                 end
#             end
#         end
#     end
#     cdedms[site,:] .= NaN
#     dms[site,:] .= NaN

#     return cdedms, dms
# end