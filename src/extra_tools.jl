function convnum2text(num)
	letters = "$num"
	res = join([(lettr == '.') ? '_' : lettr for lettr in letters])
	'_' in res && return res
	return string(res, "_0")
end


####____Leonardo code____####

function one_hot_encode(array::Array{Int8, 2})
    n = size(array, 1)
    m = size(array, 2)
    one_hot_array = zeros(Int8, n, m * 20)
    for i in 1:n
        for j in 1:m
            if array[i,j] == 21
                # Gap amino acid
                index = (j-1) * 20 + 1
                one_hot_array[i,index:index+19] = zeros(Int8, 20)
            else
                # Non-gap amino acid
                index = (j-1) * 20 + array[i,j]
                one_hot_array[i,index] = 1
            end
        end
    end
    return one_hot_array
end


function pseudocount(f, mu)
    f1=(1-mu)*f.+mu/20
    return f1
end

function filter_alignment(num_sequences::Int, matrix::Array{Int8,2})
    # Calculate the percentage of rows to keep
    percentage_to_keep = num_sequences / size(matrix, 1)

    # Calculate the number of rows to keep
    num_rows_to_keep = round(Int, size(matrix, 1) * percentage_to_keep)

    # Randomly select a subset of rows to keep
    rows_to_keep = sample(1:size(matrix, 1), num_rows_to_keep, replace=false)

    # Filter the matrix to keep only the selected rows
    filtered_matrix = matrix[rows_to_keep, :]

    return filtered_matrix
end



function matrixtofasta(path, MSA)
    n_seq = size(MSA, 1)
    FastaWriter(path, "w") do file
        for i in 1:n_seq	
            seq = MSA[i, :]
            writeentry(file, "$i", vec2string(seq) )
        end
    end
end

function find_good_target(results, target_en, target_ham)
        
    scores = []
    for i in 1:length(results[:,1])
        push!(scores, ((results[i,4]-target_en)/target_en)^2 + ((results[i,3]-target_ham)/target_ham)^2)
    end  
        
    value, idx = findmin(scores)
    println(length(scores))
    println("Score is: $value")
    print("Good choice is: ")
    println(results[idx,:])
    
    return Float64.(scores)
end


function build_seq_matrices(step_matrices::Vector{Matrix{Int8}})
    seq_matrices = Matrix{Int8}[]
    for i in 1:size(step_matrices[1], 1)
        seq_matrix = Matrix{Int8}(undef, length(step_matrices), size(step_matrices[1], 2))
        for j in 1:length(step_matrices)
            seq_matrix[j, :] = step_matrices[j][i, :]
        end
        push!(seq_matrices, seq_matrix)
    end
    return seq_matrices
end



function reshape_scores(scores_old)
    L = length(scores_old)
    scores = zeros(L, 3)
    
    for (i, tt) in enumerate(scores_old)
       scores[i, :] .= tt 
    end
    
    return scores
end

function get_fp(contacts, contacts_crystal)
    contacts_dict =  Dict( (contacts[i, 1], contacts[i, 2]) => contacts[i, 3] for i in 1:size(contacts, 1) )
    fp_tmp = []
    tp_tmp = []

    nfp = 0
    ntp = 0
    for res in contacts_dict
        if get(contacts_crystal, res[1], 0) == 0
            push!(fp_tmp, res[1])
            nfp += 1
        else
            push!(tp_tmp, res[1])
            ntp += 1
        end
    end

    fp = zeros(nfp, 2)
    tp = zeros(ntp, 2)

    for i in 1:nfp
        fp[i, 1] = fp_tmp[i][1]
        fp[i, 2] = fp_tmp[i][2]
    end
    
    for i in 1:ntp
        tp[i, 1] = tp_tmp[i][1]
        tp[i, 2] = tp_tmp[i][2]
    end
    
    return fp, tp
end


function pairwise_hamming(msa::Array{Int8,2})
    num_seqs = size(msa, 1)
    pairwise_distances = Array{Int,1}(undef, num_seqs*(num_seqs-1)÷2)
    idx = 1
    for i in 1:num_seqs-1
        for j in i+1:num_seqs
            dist = KitMSA.my_hamming(msa[i,:], msa[j,:])
            pairwise_distances[idx] = dist
            idx += 1
        end
    end
    return pairwise_distances
end