#export compute_energy_single_sequence

"""

"""
function compute_energy_single_sequence(h::Array{Float64,2},
                                        J::Array{Float64,4},
                                        S::Vector)
    N = size(h)[2]
    q = size(h)[1]
    E = 0.0
    for i = 1:N
        E -= h[S[i],i]
        for j = (i+1):N
			E -= J[S[i],S[j],i,j]
		end
	end
return E
end

