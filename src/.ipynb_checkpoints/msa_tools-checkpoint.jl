#export count_muts, count_muts_MSA

"""
	count_muts(seq1, seq2)
  	Compute the Hamming distance between sequences 'seq1' and 'seq2'.
	"seq1" and "seq2" need only to be vectors.
	The function returns an integer number: the Hamming distance. 
	
"""
function count_muts(seq1, seq2)
	ll = length(seq1)
	ll != length(seq2) && error("The two vectors do not have the same length.")
	n_muts = 0
    for i in 1:ll
        if seq1[i] != seq2[i]
            n_muts += 1
        end
    end
    return n_muts
end



"""
	count_muts_msa(MSA, seq)
	Computes the Hamming distance between all sequences in an MSA wrt 'seq'.
"""
function count_muts_msa(MSA, ref)
	L = size(MSA)[1]
	mut_vec = Array{Int16}(undef, L)
	for i in 1:L
		mut_vec[i] = count_muts(MSA[i, :], ref)
	end
	return mut_vec
end



"""
	get_pos_muts(seq1, seq2)
	Returns the positions along the sequences where they do not concide.
	Allow to identify the location of the mutations.
"""
function get_pos_muts(seq1, seq2)
	k = 0
	muts_pos = Vector{Int16}(undef, count_muts(seq1, seq2))
	for i in 1:length(seq1)
		if seq1[i] != seq2[i]
			k += 1
			muts_pos[k] = i
		end
	end
	return muts_pos
end


"""
	get_pos_muts_msa(MSA, ref_seq)
	Returns a 1D vector with the positions of all the mutations
	in the MSA wrt the reference sequences.
"""

function get_pos_muts_msa(MSA, ref_seq)
	ll = sum(count_muts_msa(MSA, ref_seq))
	pos_all_muts =  Vector{Int16}(undef, ll)
	k = 1
	for i in 1:size(MSA, 1)
	    seq = MSA[i, :]
	    pos_muts = get_pos_muts(seq, ref_seq)
		for pos in pos_muts
			pos_all_muts[k] = pos
			k +=1
	    end
	end
	return pos_all_muts
end



"""
	get_amino_muts(seq, ref_seq)
	Returns the positions along the sequences where they do not concide.
	Allow to identify the location of the mutations.
"""
function get_amino_muts(seq, ref_seq)
	k = 0
	muts_amino = Vector{Int16}(undef, count_muts(seq, ref_seq))
	for i in 1:length(seq)
		if seq[i] != ref_seq[i]
			k += 1
			muts_amino[k] = seq[i]
		end
	end
	return muts_amino
end



"""
	get_amino_muts_msa(MSA, ref_seq)
	Returns a 1D vector with the aminoacids(as numbers) of all the mutations
	in the MSA wrt the reference sequence.
"""

function get_amino_muts_msa(MSA, ref_seq)
	ll = sum(count_muts_msa(MSA, ref_seq))
	amino_all_muts =  Vector{Int16}(undef, ll)
	k = 1
	for i in 1:size(MSA, 1)
	    seq = MSA[i, :]
	    amino_muts = get_amino_muts(seq, ref_seq)
		for amino in amino_muts
			amino_all_muts[k] = amino
			k +=1
	    end
	end
	return amino_all_muts
end





"""
	remove_gapped_seqs(msa_file, max_gaps)
	Writes in the same folder as "msa_file" the same MSA
	whose sequences with more than max_gaps have been removed.
	Sequences are renamed in increasing order.
"""

bool_gaps(seq, max_gaps) = sum([1 for l in seq if ((l == '-') || (l == 21))]) > max_gaps ? true : false


function remove_gapped_seqs(msa_file, max_gaps)
	dir, file = splitdir(msa_file)
	split_f = split(file, ".")
	l = length(split_f)
	insert!(split_f, l, "max.$(max_gaps).gaps")
	file_out = joinpath(dir, join(split_f, "."))
	f = FastaReader(msa_file)
	i = 1
	for (desc, seq_string) in f
		if !bool_gaps(seq_string, max_gaps)
			desc = i == 1 ? desc : "$i"
			writefasta(file_out, [(desc, seq_string)], "a")
			i += 1
		end
	end
	close(f)
end


"""
	detect_sequencial_muts(seq, ref_seq, l_shift)
	Returns 'true' if the sequence has an equal number of more
	mutations wrt the ref_seq that are sequencial.
	The function excludes all parts of the sequence
	that contain gaps, that is, they are not counted as mutations.
"""

function detect_sequencial_muts(seq, ref_seq, l_shift)
	pos_muts = get_pos_muts(seq, ref_seq)
	n_muts = length(pos_muts)
    for i in l_shift : n_muts
		bool_shift = ( pos_muts[i] - pos_muts[i + 1 - l_shift] ) == (l_shift -1)
		bool_gap = bool_gaps(seq[ (pos_muts[i] - l_shift +1) : pos_muts[i]] , 0)
        (bool_shift && !bool_gap) && return true
    end
    return false
end


function remove_shift_MSA(msa_file, shift_length, ref_seq)
	file_out = "$(msa_file).no.shift"
	f = FastaReader(msa_file)
	i = 1
	for (desc, seq_string) in f
		muts = get_pos_muts(string2vec(seq_string), ref_seq)
		desc = i == 1 ? desc : "$i"
		!detect_sequencial_muts(muts, shift_length) && writefasta(file_out, [(desc, seq_string)], "a")
		i += 1
	end
	close(f)
end


"""
	remove_lowcase_MSA(msa_file)
	It does what it says.
"""

function remove_lowcase(seq)
	return join([lett for lett in seq if !islowercase(lett)])
end

function remove_lowcase_MSA(msa_file)
	file_out = "$(msa_file).no.ins"
	f = FastaReader(msa_file)
	for (desc, seq_string) in f
		seq = remove_lowcase(seq_string)
		writefasta(file_out, [(desc, seq)], "a")
	end
	close(f)
end
