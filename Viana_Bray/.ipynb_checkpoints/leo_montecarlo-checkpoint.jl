function initialize_seq(L::Int)

    ### Initialize the spin configuration to a random one
    S = zeros(Int,L)
    for i = 1:L
        if rand() > 0.5
            S[i] = 1
        else
            S[i] = -1
        end
    end
    return S
end


function graph_fixed_mean_conn(L::Int, C::Int, var_J::T, var_h::T) where {T}
    
    p = C/(L-1)
    
    ### Initialize seed RNG
    RNGseed = 10
    Random.seed!(RNGseed)


    ### Initialize fields & couplings
    FieldDistribution = Laplace(zero(T),var_h) 
    h=rand(FieldDistribution, L)  

    J = zeros(Float64,L,L)
    for i = 1:L
        for j = i+1:L
            if rand() < p
                Jvalue = rand(Normal(zero(T),var_J))
                J[i,j] = Jvalue
                J[j,i] = Jvalue
            end
        end
    end

    for i in 1:L
        for j in 1:L
            @assert J[i,j] == J[j,i]
        end
    end
    
    return h,J
end

function Delta_energy(idx::Int, h::Array{T,1}, J::Array{T,2}, seq::Array{Int,1}, L::Int) where {T}

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

    
    
    _ΔE = h[idx]
    for j in 1:L#findall(x -> x != 0, view(J,idx,:))
        _ΔE += J[idx,j]*seq[j] 
    end
    ΔE = 2*seq[idx]*_ΔE
    
    return ΔE
end


struct ChainMc
    seq::Array{Int,1}
    L::Int
    generator::Xoshiro
end

function ChainMc(seq::Array{Int,1}, generator::Xoshiro)
    L = size(seq,1)
    ChainMc(seq, L, generator)
end


function random_gensMc(num_generators::Int) 
    rng_array = []
    for seed in 1:num_generators
        push!(rng_array, Random.Xoshiro(seed))
    end
    return rng_array
end

function Metrop_move!(seq::Array{Int,1}, site::Int, h::Array{T,1}, J::Array{T,2}, β::Float64, L::Int) where {T}
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
    #println("Pre_mut hamming: ", sum(S .!= seq)/L)
    
    ΔE = Delta_energy(site, h, J, seq, L)
    #println("ΔE: ", ΔE)
    
    if ΔE < 0
        #println("negative mut accepted")
        #println("Before flip: ", seq[site])
        seq[site] *= -1
        #println("After flip: ", seq[site])
    elseif rand() < exp(-β*ΔE)
        #println("positive mut accepted")
        #println("Before flip: ", seq[site])
        seq[site] *= -1
        #println("After flip: ", seq[site])
    else
        #println("Mutation rejected")
    end
    
    #println("Post_mut hamming: ", sum(S .!= seq)/L)
    #println("   ")
    
end



function MCMC_Sampling(S::Array{Int,2}, h::Array{T,1}, J::Array{T,2}; β = 1., rand_init = false, N_steps::Int = 100, N_chains::Int = 10, N_points::Union{Int, Nothing} = nothing) where {T}
    
    
    L = length(h)
    TT = eltype(h)
    β = T(β)
    
    rng = random_gensMc(N_chains)
    if rand_init == true
        println("Random Initialization")
        chains = [ChainMc(initialize_seq(L), rng[n]) for n in 1:N_chains]
    else
        println("Wts Initialization")
        #chains = [ChainMc(S, rng[n]) for n in 1:N_chains]
        chains = [deepcopy(S[i,:]) for i in 1:N_chains]
    end
    
    
    
    if N_points !== nothing 
        points = unique([trunc(Int,10^y) for y in range(log10(1), log10(N_steps), length=N_points)])
        step_msa = [zeros(Int, (N_chains, L)) for i in 1:length(points)]
    end
    
    count = 0
    
    for t in 1:N_steps
        @tasks for n in 1:N_chains
            Metrop_move!(chains[n], rand(rng[n], 1:L), h, J, β, L)
        end
        
        if (N_points !== nothing) && (t in points)
            count += 1
            @tasks for n in 1:N_chains
                for i in 1:L
                    step_msa[count][n,i] = chains[n][i]
                end
            end
        end
    end
    
    if N_points !== nothing 
        return (step_msa = step_msa, times = points)
    else
        msa = zeros(N_chains, L)
        @tasks for n in 1:N_chains
            for i in 1:L
                msa[n,i] = chains[n][i]
            end
        end
        return (msa = msa)
    end
end


    
    

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

function H_distance(seq::Vector, msa::Matrix)
    ### Function computing the Hamming distance between two vectors

    L = length(seq)
    @assert L == size(msa,2) "Error: the two vectors do not have the same length"
    M = size(msa,1)
    d = zeros(M)
    for m in 1:M
        for i in 1:L
            if seq[i] != msa[m,i]
                d[m] += 1
            end
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


function ham_dist(seq1::Array{Int,1}, seq2::Array{Int,1}, L::Int)
    return sum(seq1 .!= seq2) / L
end


function ham_dist(seq1::Array{Int,1}, seq2::Array{Int,1})
    L = length(seq1)
    return ham_dist(seq1, seq2, L)
end

function ham_dist(seq1::Array{Int,1}, msa::Array{Int,2})
    M,L = size(msa)
    
    return [ham_dist(seq1, msa[i,:],L) for i in 1:M][:]
end

function ham_dist(msa1::Array{Int,2}, msa2::Array{Int,2})
    M,L = size(msa1)
    return [ham_dist(msa1[i,:], msa2[i,:],L) for i in 1:M][:]
end
    
