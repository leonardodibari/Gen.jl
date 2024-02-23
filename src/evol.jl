# export evol_msa_fix_steps

##Requires LinearAlgebra


"""
	cond_proba(k, mutated_seq, h, J, prob, T = 1)

	"k": position along an amino acid chain
	"mutated_seq": sequence of amino acid of interest
	"h", "J": parameters of DCA Hamiltonian
	"prob": empty vector of length 
	"T": temperature of DCA

	Returns a vector of length 21 with the conditional probability of having
	aminoacid " 1 2 3 ... 21" in that sequence context.
"""

function cond_proba(k, mutated_seq, h, J, prob, q, N, T = 1)
	@inbounds for q_k in 1:q
		log_proba = h[q_k, k]
 		for i in 1:N
			log_proba += J[mutated_seq[i], q_k ,i, k]
		end
		prob[q_k] = exp(log_proba/T)
        
	end
	return normalize(prob,1)
end




"""
	evol_seq_fix_steps(ref_seq, MC_steps, h, J, N, pop, prob, T = 1)


	Returns a sequence obtained by Gibbs sampling after "MC_steps" steps.
	In input the reference sequence, and all necessary paramters.
	Optimized to get "pop", "prob" as imput from a "new_msa_xxx" function.

	"ref_seq": amino acid vector of initial sequence
	"MC_steps" : number of Monte Carlo steps performed
	"N" : length of the evolved protein
	"h", "J": parameters of DCA Hamiltonian
	"prob", "prob": utilities to run code faster
	"T": temperature of DCA
"""

function evol_seq_fix_steps(ref_seq, MC_steps, h, J, q, N, pop, prob, T = 1)
	mutated_seq = deepcopy(ref_seq)
	@inbounds for steps in 1: MC_steps
		pos_mut = rand(1:N)
		mutated_seq[pos_mut] = sample(pop, ProbabilityWeights(cond_proba(pos_mut, mutated_seq, h, J, prob, q, N, T)))
	end
	return mutated_seq
end



function evol_seq_fix_steps(ref_seq, MC_steps, h, J, q; N = length(ref_seq), T = 1, pop = collect(Int16, 1:q), prob = Vector{Float64}(undef, q))
	mutated_seq = deepcopy(ref_seq)
	@inbounds for steps in 1: MC_steps
		pos_mut = rand(1:N)
		mutated_seq[pos_mut] = sample(pop, ProbabilityWeights(cond_proba(pos_mut, mutated_seq, h, J, prob, q, N, T)))
	end
	return mutated_seq
end






"""
	evol_msa_fix_steps_persistent(output_file, seed_seq, MC_steps, n_seq, h, J;T = 1, prot_name = "TEM-1")

	Writes a MSA obtained by Gibbs sampling.

	INPUT:
	"output_file": file where to print MSA in fasta format
	"seed_seq": amino acid vector of initial sequence
	"MC_steps": number of Monte Carlo steps performed to evolve each seq
	"n_seq": number of sequences in the MSA
	"h", "J": parameters of DCA Hamiltonian
	"T": temperature of DCA
	"prot_name": name of the original protein sequence
"""


function evol_msa_fix_steps_persistent(output_file, seed_seq, MC_vec, n_seqs, h, J, index; T = 1, prot_name = "PSE-1", gaps = true)
    q = gaps == true ? 21 : 20
    N = length(seed_seq)
    pop = collect(Int16, 1:q)
    prob = Vector{Float64}(undef, q)
    T_letter = convnum2text(T) 
    for nn in 1:n_seqs
        seq = seed_seq
        steps = MC_vec[1]
        for (i, MC_steps) in enumerate(MC_vec)  
            file_name = joinpath(output_file,
                "$(prot_name)_silico_seqs_$(n_seqs)_T_$(T_letter)_MCsteps_$(MC_steps).$(index)")     
            FastaWriter(file_name, "a") do file
                seq = evol_seq_fix_steps(seq, steps, h, J, N, pop, prob, T)
                writeentry(file, "$nn |evolved from $prot_name with Gibbs Sampling | $MC_steps MC steps,T = $(T)", vec2string(seq))
            end
            if i != length(MC_vec)
                steps = MC_vec[i + 1] - MC_vec[i]
            end 
        end
    end
end



function evol_seq_keep_history(output_file, seed_seq, MC_vec, h, J, index; T = 1, prot_name = "PSE-1", gaps = true)
    q = gaps == true ? 21 : 20
    N = length(seed_seq)
    pop = collect(Int16, 1:q)
    prob = Vector{Float64}(undef, q)
    T_letter = convnum2text(T) 
    seq = seed_seq
    steps = MC_vec[1]

    file_name = joinpath(output_file,
        "$(prot_name)_silico_evol_T_$(T_letter).$(index).fasta")

    FastaWriter(file_name, "a") do file

        for (i, MC_steps) in enumerate(MC_vec)

            seq = evol_seq_fix_steps(seq, steps, h, J, N, pop, prob, T)
            writeentry(file, 
            "$i |evolved from $prot_name with Gibbs Sampling | $MC_steps MC steps, T = $(T)",                     vec2string(seq)) 

            if i != length(MC_vec)
                steps = MC_vec[i + 1] - MC_vec[i]
            end   
        end
    end

end




function resample_MSA(file_in, file_out, seqs)  
    MSA = readfasta(file_in)
    M = size(MSA)[1]
    index_v = sample(collect(1:M), seqs, replace = false)
    for j in 1:seqs
        index = index_v[j]
        desc, seq = MSA[index]
        split_d = split(desc)
        rejoin = join(split_d[2:end], " ")
        desc = "$j $rejoin"
        writefasta(file_out, [(desc, seq)], "a")
    end
end


function resample_MSA(file_in, seqs_vec)
    MSA = readfasta(file_in)
    M = size(MSA)[1]
    dir, file = splitdir(file_in)
    split_f = split(file, "_")
    for n_seqs in seqs_vec
        split_f[4] = "$n_seqs"
        file_out = joinpath(dir, join(split_f, "_"))
        index_v = sample(collect(1:M), n_seqs, replace = false)
        for j in 1:n_seqs
            index = index_v[j]
            desc, seq = MSA[index]
            split_d = split(desc)
            rejoin = join(split_d[2:end], " ")
            desc = "$j $rejoin"
            writefasta(file_out, [(desc, seq)], "a")
        end
    end
    println("Resampled!")
end



function sample_BM(h, J, n_steps, n_seqs, file_out)
    q = size(h, 1)
    N = size(h, 2)
    for i in 1:n_seqs
	ref_seq = rand(1:q, N)
	seq = evol_seq_fix_steps(ref_seq, n_steps, h, J, q; N = N, T = 1, 	
	pop = collect(Int16, 1:q), prob = Vector{Float64}(undef, q))
	writefasta(file_out, [("$i", vec2string(seq))], "a")
    end
end


function sample_from_MSA(path_MSA, h, J, n_steps, file_out)
    q = size(h, 1)
    N = size(h, 2)
    MSA = fasta2matrix(path_MSA)
    n_seqs = size(MSA, 1)
    for i in 1:n_seqs
	ref_seq = MSA[i, :]
	seq = evol_seq_fix_steps(ref_seq, n_steps, h, J, q; N = N, T = 1, 	
	pop = collect(Int16, 1:q), prob = Vector{Float64}(undef, q))
	writefasta(file_out, [("$i", vec2string(seq))], "a")
    end
end


####____Leonardo code___####

function get_accessible_nucleo_muts_DNA_det_bal(old_codon, nucleo_pos)
    old_codon = [string(old_codon[i]) for i in 1:3 ]
	codon_list = Vector{AbstractString}(undef, 4)
	new_codon = deepcopy(old_codon)
	for (j, nucl) in enumerate(["A", "C", "G", "T"]) 
		new_codon[nucleo_pos] = nucl
		codon_list[j] = join(new_codon)
	end
    
	amino_list = get.(Ref(cod2amino), codon_list, 0)
	amino_list = unique!(filter!(aa->aa != 21, amino_list))

	return amino_list, codon_list
end

function nucleo_cond_proba_DNA_gibbs(k, codon_list, mutated_seq, h, J,N,  T = 1)
	prob = zeros(length(codon_list))
	for (index, codon_k) in enumerate(codon_list)
        q_k = cod2amino[codon_k]
		log_proba = h[q_k, k]-log(length(amino2cod[q_k]))
 		for i in 1:N
			log_proba += J[mutated_seq[i], q_k ,i, k]
		end
		prob[index] = exp(log_proba/T)
	end
	return normalize(prob,1)
end


function evol_seq_fix_steps_DNA_gibbs_nucleo(ref_seq, MC_steps, h, J, N, T = 1)
	mutated_seq = deepcopy(ref_seq)
	non_gapped_pos = [pos for (pos, amino) in enumerate(ref_seq.Amino) if amino != 21]
	@inbounds for steps in 1: MC_steps
        pos_mut = rand(non_gapped_pos)

		old_codon = mutated_seq.DNA[pos_mut]
        
        nucleo_pos = rand(1:3)

		amino_list, codon_list = get_accessible_nucleo_muts_DNA_det_bal(old_codon, nucleo_pos)
        
		new_codon = sample(codon_list, ProbabilityWeights(nucleo_cond_proba_DNA_gibbs(pos_mut, codon_list, mutated_seq.Amino, h, J,N,  T)))
        #println(new_codon)
        aa = cod2amino[new_codon]
        #println(mutated_seq.Amino[pos_mut])
		mutated_seq.DNA[pos_mut] = new_codon	
		mutated_seq.Amino[pos_mut] = aa
        #println(aa)
        #println(mutated_seq.Amino[pos_mut])
        #println("mutazione")
	end
    #println(KitMSA.my_hamming(ref_seq.Amino,mutated_seq.Amino))
    #println(ref_seq.Amino)
    #println(mutated_seq.Amino)
    
	return mutated_seq.Amino
end

function evol_seq_fix_steps_DNA_metropolis_nucleo(seed_seq, steps, h, J, N, T)
    mutated_seq = deepcopy(ref_seq)
	non_gapped_pos = [pos for (pos, amino) in enumerate(ref_seq.Amino) if amino != 21]
    @inbounds for steps in 1: MC_steps
        pos_mut = rand(non_gapped_pos)

		old_codon = mutated_seq.DNA[pos_mut]
        
        nucleo_pos = rand(1:3)
        
        
        
    	amino_list, codon_list = get_accessible_nucleo_muts_DNA_det_bal(old_codon, nucleo_pos)
        new_codon = sample(codon_list)#is this sure?
        acceptance_prob = rand(0:1)
        
        
        
		#new_codon = sample(codon_list, ProbabilityWeights(nucleo_cond_proba_DNA_metropolis(pos_mut, codon_list, mutated_seq.Amino, h, J,N,  T)))
        #println(new_codon)
        aa = cod2amino[new_codon]
        #println(mutated_seq.Amino[pos_mut])
		mutated_seq.DNA[pos_mut] = new_codon	
		mutated_seq.Amino[pos_mut] = aa
        #println(aa)
        #println(mutated_seq.Amino[pos_output])
        #println("mutazione")
	end
    #println(KitMSA.my_hamming(ref_seq.Amino,mutated_seq.Amino))
    #println(ref_seq.Amino)
    #println(mutated_seq.Amino)
    
	return mutated_seq.Amino
end


#version for the grid search 

function nucleo_evolMSA2(sampler_type, output_path::AbstractString, seed_seq; h, J, steps::Integer = 10, nseq::Integer = 100, T::Real = 1, wt_name::AbstractString = "unknown wt") 
    N = length(seed_seq.Amino)
    println("new_det_bal_$(wt_name)_silico_seqs_$(nseq)_T_$(T)_MCsteps_$(steps).$(sampler)DNA")
	FastaWriter(output_path, "a") do file
        if sampler_type == "gibbs"
            for i in 1:nseq	
                seq = evol_seq_fix_steps_DNA_gibbs_nucleo(seed_seq, steps, h, J, N, T)
                writeentry(file, "sampler: $sampler | $i | original wt: $wt_name | $steps MC steps | T = $(T)", vec2string(seq))	
                if(i%1000==0)
                    println("$i sequences generated")
                end
            end
        elseif sampler_type == "metropolis"
            for i in 1:nseq    
                seq = evol_seq_fix_steps_DNA_metropolis_nucleo(seed_seq, steps, h, J, N, T)
                writeentry(file, "sampler: $sampler | $i | original wt: $wt_name | $steps MC steps | T = $(T)", vec2string(seq))    
                #if(i%5000==0)
                    #println("$i sequences generated")
                #end
            end
        else
            throw(ArgumentError("Invalid sampler type: $sampler_type"))
        end
	end	
end


####
#MAIN FUNCTION in this version I extract the parameters from outside
####

function nucleo_evolMSA2(sampler_type, output_path::AbstractString, wt_path::AbstractString; h, J, steps::Integer = 10, nseq::Integer = 100, T::Real = 1, wt_name::AbstractString = "unknown wt") 
	println("1")
    for file in [wt_path]
	    !isfile(file) && error("Error: the file \"$(file)\" does not exist. Please check the spelling or the folder path.")
	end
    
    file = open(output_path, "w")
	println("2")
	steps < 1 && throw(DomainError(steps, "'steps' must be a positive integer."))
	nseq < 1 && throw(DomainError(nseq, "'nseq' must be a positive integer."))
	T <= 0 && throw(DomainError(T, "'T' must be a positive real number."))
	println("3")
	seq = join(readdlm(wt_path, skipstart = 1))
	L = Int64(length(seq)/3)
	DNA_seq = [seq[((i-1)*3 +1):(i*3)] for i in 1:L]
	amino_seq = [cod2amino[codon] for codon in DNA_seq]
	seed_seq = SeqToEvolve(amino_seq, DNA_seq)
	N = length(seed_seq.Amino)
    println("new_det_bal_$(wt_name)_silico_seqs_$(nseq)_T_$(T)_MCsteps_$(steps).$(sampler)DNA")
	FastaWriter(output_path, "a") do file
        if sampler_type == "gibbs"
            for i in 1:nseq	
                seq = evol_seq_fix_steps_DNA_gibbs_nucleo(seed_seq, steps, h, J, N, T)
                writeentry(file, "sampler: $sampler | $i | original wt: $wt_name | $steps MC steps | T = $(T)", vec2string(seq))	
                if(i%10000==0)
                    println("$i sequences generated")
                end
            end
        elseif sampler_type == "metropolis"
            for i in 1:nseq    
                seq = evol_seq_fix_steps_DNA_metropolis_nucleo(seed_seq, steps, h, J, N, T)
                writeentry(file, "sampler: $sampler | $i | original wt: $wt_name | $steps MC steps | T = $(T)", vec2string(seq))    
                #if(i%5000==0)
                 #   println("$i sequences generated")
                #end
            end
        else
            throw(ArgumentError("Invalid sampler type: $sampler_type"))
        end
	end	
end


function grid_search(rounds, temps, sampler_type, filename::AbstractString, seed_seq; h, J, nseq::Integer = 100, wt_name::AbstractString = "unknown wt")
    
    results = []
    
    wt_ref = seed_seq.Amino
    
    for steps in rounds
        for T in temps
            
            output_path = joinpath("..", "data_matteo", "my_project", "det_bal_$(wt_name)_silico_seqs_$(nseq)_T_$(T)_MCsteps_$(steps).$(sampler)DNA")
            nucleo_evolMSA2(sampler, output_path, seed_seq; h, J, steps, nseq, T, wt_name)
            
            new_MSA = Int8.(fasta2matrix(output_path))
            
            muts_new = count_muts_msa(new_MSA, wt_ref)
            en_new = [KitMSA.energy(h, J, new_MSA[i, :]) for i in 1:size(new_MSA, 1)]
            mean_muts_new = round.(mean(muts_new); digits=2)
            mean_en_new = round.(mean(en_new); digits=2)
            
            info = [steps, T, mean_muts_new, mean_en_new]
            push!(results, info)
            
        end
    end
    
    writedlm(filename, results)
end