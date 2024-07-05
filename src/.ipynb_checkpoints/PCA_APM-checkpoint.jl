function perform_pca(MSA)

    W, Meff = compute_weights(Int8.(MSA'), 22, 0.2)
    q = 21
    M, L = size(MSA)
    seqs = Int8.(MSA')
    
    println("Computing statistics...")
    
    one_hot_seqs = zeros(M,L*(q-1))
    for m in 1:M    # Performs one-hot encoding (ignoring gaps, why?)
        for i in 1:L
            if seqs[i,m] < q
                one_hot_seqs[m,(i-1)*(q-1)+seqs[i,m]] = 1.0
            end
        end
    end

    pij = zeros(L*(q-1),L*(q-1))
    pi = zeros(L*(q-1),)

    for m in 1:M
        for i in 1:L
            if seqs[i,m] < q
                pi[(i-1)*(q-1) + seqs[i,m]] += W[m]
                for j = i:L
                    if seqs[j,m] < q
                        pij[(i-1)*(q-1) + seqs[i,m], (j-1)*(q-1) + seqs[j,m]] += W[m]
                        pij[(j-1)*(q-1) + seqs[j,m], (i-1)*(q-1) + seqs[i,m]] += W[m] 
                    end
                end
            end
        end
    end

    pij /= Meff
    pi /= Meff
    cij = pij - pi*transpose(pi)

    println("Performing eig factorization...")
    F = eigen(cij)
    println("Largest eigenvalues: ", F.values[end-1:end])
    PC1 = F.vectors[:,end]
    PC2 = F.vectors[:,end-1]

    proj_seqs = (one_hot_seqs * PC1, one_hot_seqs * PC2)

    println("Done")
    return proj_seqs, (PC1, PC2)
end

function get_projection(PC,  seq::Vector{Int}; q::Int=21)

    L = length(seq)
	one_hot_sample = zeros(1,L*(q-1))
	
	@assert length(PC[1]) == L*(q-1)
    # println("Encoding sample...")
    for i in 1:L
        if seq[i] < q # exclude gaps (whatever learning method you used)
            one_hot_sample[1,(i-1)*(q-1)+ seq[i]] = 1.0
        end
    end
	# println("Projecting sample...")
	proj_sample = (one_hot_sample * PC[1], one_hot_sample * PC[2])	
	# println("Done")
	return proj_sample
end

function plot_in_pc_space(x::Vector, y::Vector;
    nbins::Int = 200, label::String="")
    #gr()
    p = histogram2d(x,y, nbins=nbins, title=label, xlabel = "PC1", ylabel = "PC2", aspect_ratio = :equal, c=cgrad(:thermal, rev=true))
    return p
end

function distance_PC(p1::Vector, p2::Vector)
    L = length(p1) 
    @assert L == length(p2)
    distance  = 0
    for i in 1:L
        distance += (p1[i] - p2[i])^2
    end
    return sqrt(distance)
end