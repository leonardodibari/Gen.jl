function compare_seq(a::Array{Int8,1}, b::Array{Int8,1})
    if a == b
        return 0
    else
        return findall(a .!= b)[1]
    end
end

function find_positions(msa)
    
    return [compare_seq(msa[j-1,:], msa[j,:]) for j in 2:length(msa[:,1])]
        
end

function get_entropy(f; q = 21)
    N=length(f[1,:])
    entr = zeros(Float64, N)
    for i in 1:N
        for a in 1:q
            if(f[a,i]>0)
                entr[i]-=f[a,i]*log(f[a,i])
            end
        end
    end
    
    return entr / log(2)
end

function proba_DNA_gibbs_without_deg(k, mutated_seq, h, J, N; q = 21, T = 1)
	prob = zeros(q)
	for i in 1:q
        q_k = i
		log_proba = h[q_k, k]
 		for j in 1:N
			log_proba += J[mutated_seq[j], q_k , j, k]
        end
		prob[i] = exp(log_proba/T)
	end
	return normalize(prob,1)
    
end

function cont_dep_entr_without_deg(background, h, J; q =21, T =1)
    
    N = length(background)
    
    prob = hcat([ProbabilityWeights(proba_DNA_gibbs_without_deg(pos_mut, background, h, J, N, T=T, q=q)) for pos_mut in 1:N]...)

    return get_entropy(prob, q = q)[:]
end

function write_cde_of_chains(folder, out_path, mask, h, J, n)
    fasta_files = filter(file -> endswith(file, ".mixedDNA"), readdir(folder))
    res = []
    for i in 1:n
        chain = Int8.(fasta2matrix(joinpath(folder_path, fasta_files[i])))[mask,:]
        cde_msa = hcat([cont_dep_entr_without_deg(chain[step,:], h, J, q= 21) 
                for step in 1:length(chain[:,1])]...)'
        push!(res, vec(cde_msa))
        if i %10 == 0
            println(i)
        end
    end
    writedlm(out_path, hcat(res...))
    return hcat(res...)
end

function cde_1site(site, background, h, J; q = 21,  T = 1)
  
    N = length(background)
    prob = ProbabilityWeights(proba_DNA_gibbs_without_deg(site, 
            background, h, J, N, T = T, q = q))
    return get_entropy(prob, q = q)
end


function cde_chain(chain, h, J)
    
    res = zeros(size(chain))
    
    res[1,:] = cont_dep_entr_without_deg(chain[1,:], h, J, q= 21)[:] 
    
    for i in 2:size(chain, 1)
        if matrices[i,:] == matrices
            res[i,:] = res[i-1, :]
        else 
            res[i,:] = cont_dep_entr_without_deg(chain[i,:], h, J, q= 21)[:]
        end
    end
    
    return res
end

function CIE(msa; q = 21)
    L = size(msa,2)
    f = reshape(DCAUtils.compute_weighted_frequencies(Int8.(msa'), q+1, 0.2)[1], (q, L))
    return get_entropy(f, q = q)
end



function proba_DNA_gibbs_masked(k, sites, masked_seq, h, J; q = 21, T = 1)
	prob = zeros(q)
	for i in 1:q
        q_k = i
		log_proba = h[q_k, k]
        n = 0
 		for j in sites
            n += 1
            log_proba += J[masked_seq[n], q_k , j, k]
        end
		prob[i] = exp(log_proba/T)
	end
	return normalize(prob,1)
    
end

function proba_DNA_gibbs_masked_no_norm(k, sites, masked_seq, h, J; q = 21, T = 1)
	prob = zeros(q)
	for i in 1:q
        q_k = i
		log_proba = h[q_k, k]
        n = 0
 		for j in sites
            n += 1
            log_proba += J[masked_seq[n], q_k , j, k]
        end
		prob[i] = exp(log_proba/T)
	end
	return prob
    
end

function proba_DNA_gibbs_field(k, h, J; q = 21, T = 1)
	prob = zeros(q)
	for i in 1:q
        q_k = i
		log_proba = h[q_k, k]
		prob[i] = exp(log_proba/T)
	end
	return normalize(prob,1)
    
end

function cde_field(pos_mut, h, J; q =21, T =1)
     
    prob = ProbabilityWeights(proba_DNA_gibbs_field(pos_mut, h, J, T=T, q=q)) 

    return get_entropy(prob, q = q)[:]
end

function cde_masked(sites, masked_seq, h, J; q =21, T =1)
    
    prob = hcat([ProbabilityWeights(proba_DNA_gibbs_masked(pos_mut, sites, masked_seq, h, J, T=T, q=q)) for pos_mut in 1:N]...)

    return get_entropy(prob, q = q)[:]
end

function cde_masked(pos_mut, sites, masked_seq, h, J; q =21, T =1)

    prob = ProbabilityWeights(proba_DNA_gibbs_masked(pos_mut, sites, masked_seq, h, J, T=T, q=q))
    
    return get_entropy(prob, q = q)[:]
end



function TT(Z::Array{Int8,2},cde_nat::Array{Float64,2},
        h::Array{Float64,2},J::Array{Float64,4}, W::Array{Float64,1})
    N, L = size(Z)
    T = zeros(Float64,L,L,21)
    mcde = mean(cde_nat, weights(W), dims = 1)[:]

    for site in 1:L
        for s in 1:N
            CDE_mysite = cde_1site(site, Z[s,:], h, J, q = 21)[1] 
            for n in 1:L
                if n != site
                    T[site,n,Z[s,n]] += (CDE_mysite - mcde[site]) * W[s]
                end
            end
        end
        T[site,:,:] ./= sum(W)
    end
    return T
end


function TT_singlesite(site::Int, Z::Array{Int8,2},cde_nat::Array{Float64,2},
        h::Array{Float64,2},J::Array{Float64,4}, W::Array{Float64,1})
    N, L = size(Z)
    T = zeros(Float64,L,21)
    mcde = mean(cde_nat[:,site], weights(W), dims = 1)[1]
    #println(mcde)

    for s in 1:N
        CDE_mysite = cde_1site(site, Z[s,:], h, J, q = 21)[1] 
        #println(CDE_mysite)
        for n in 1:L
            if n != site
                T[n,Z[s,n]] += (CDE_mysite - mcde) * W[s]
            end
        end
    end
    T ./= sum(W)
    return T
end


function TTcond(site::Int, freq::Array{Float64, 2}, n_aminos::Int, Z::Array{Int8,2}, cde_nat::Array{Float64,2},
        h::Array{Float64,2},J::Array{Float64,4}, W::Array{Float64,1})
    
    L = size(Z,2)
    tcond = zeros(Float64, L, 21, L, 21)
    
    aminos = sortperm(freq[:,19], rev = true)[1:n_aminos]
    
    for aa in aminos
        idxs = Z[:,site] .== aa
        Z_red = Z[idxs, :]
        cde_red = cde_nat[idxs, :]
        _w = W[idxs]
        tcond[site, aa, : ,:] = TT_singlesite(site, Z_red, cde_red, h, J, _w)
    end   

    return tcond
end


function CDE_masked_back(site::Int, tt::Array{Float64, 2}, nat_msa::Array{Int8,2},
        cde_nat::Array{Float64,2}, title, filename)
    n_seq = size(nat_msa,1)
    res = zeros(76, n_seq)  
    ord = sortperm([maximum(abs.(tt[i,:])) for i in 1:size(tt,1)], rev = true)
    for idx in 1:n_seq
        res[:, idx] = ([cde_masked(site, ord[1:i], nat_msa[idx,ord[1:i]], h, J)[1] 
            for i in 1:76] .- cde_nat[idx, site]);
    end
    
    for i in 1:n_seq
        plt.plot(res[:,i], alpha = 0.01, color = "black")
    end
    plt.plot(mean(res, dims = 2),linestyle = "--", color = "red", label = "mean")
    plt.plot([0,76], [0.3,0.3], linestyle = "--", color = "green", label = "tresh")
    plt.plot([0,76], [-0.3,-0.3], linestyle = "--", color = "green")
    plt.title(title)
    plt.legend()
    savefig(filename)
    
    return res
end 

function CIE(msa; q = 21)
    L = size(msa,2)
    f = reshape(DCAUtils.compute_weighted_frequencies(Int8.(msa'), q+1, 0.2)[1], (q, L))
    return get_entropy(f, q = q)
end
    
    