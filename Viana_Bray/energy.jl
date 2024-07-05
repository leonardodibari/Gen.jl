#export compute_energy_single_sequence

"""

"""
function compute_energy(h::Array{Float64,1},
                                        J::Array{Float64,2},
                                        S::Array{Int,1})
    N = length(h)
    E = 0.0
    for i = 1:N
        E -= h[i]*S[i]
        for j = i+1:N
            E -= J[i,j]*S[i]*S[j]
        end
	end
    return E
end

function compute_energy(h::Array{Float64,1},
                                        J::Array{Float64,2},
                                        msa::Array{Int,2})
    M,L = size(msa)
    ens = [compute_energy(h,J,msa[i,:]) for i in 1:M] 
    return ens
end

