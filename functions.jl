using PyCall
using LinearAlgebra
using Random
using PyPlot

export conductivity

# Run the Python script
py"""
exec(open("parameters.py").read())
"""

# Access Python variables in Julia
L = py"L"
eta_factor = py"eta_factor"
u = py"u"
l0 = py"l0"
T = py"T"
ef = py"ef"
h_cut = py"h_cut"
vf = py"vf"
sx2 = py"sx2"
k_space_size = py"k_space_size"
N_i = py"N_i"

# Gaussian correlation function
function gaussian_corr(q, u, l0)
    return u * exp(-q^2 * l0^2 / 2)
end

# Thomas-Fermi screening function
function thomas_fermi(q, u, l0)
    return u / (q + l0^(-1))
end

pot_function = thomas_fermi

# Fermi-Dirac distribution
function fermi_dirac(x, T=T, ef=ef)
    if T != 0
        return 1 / (1 + exp((x - ef) / T))
    elseif x > ef
        return 0
    elseif x == ef
        return 0.5
    else
        return 1
    end
end

function conductivity(L=L, eta_factor=eta_factor, R_I=[0 1;1 2;2 3], u=u, l0=l0)
    
    factor = -1im * 2 * π * h_cut^2 / L^2 * vf^2
    eta = eta_factor * vf * 2 * π / L

    lamda = 20 * π / L
    k_vec = range(-lamda, lamda, length=k_space_size)

    potential = zeros(Complex{Float64}, (k_space_size^2, k_space_size^2))
    H0 = zeros(Complex{Float64}, (k_space_size^2, k_space_size^2))

    k_cartesian = [(kx, ky) for kx in k_vec, ky in k_vec]

    for (i, (kx1, ky1)) in enumerate(k_cartesian)
        for (j, (kx2, ky2)) in enumerate(k_cartesian)
            kx = kx1 - kx2
            ky = ky1 - ky2
            kk = sqrt(kx^2 + ky^2)
            for I in 1:size(R_I, 2)
                potential[i, j] += exp(1im * (kx * R_I[1, I] + ky * R_I[2, I])) * pot_function(kk, u, l0)
            end
        end
    end

    potential /= L^2

    for i in 1:k_space_size^2-1
        H0[i+1, i] = vf * (k_vec[mod(i, k_space_size)+1] + 1im * k_vec[div(i, k_space_size)+1])
        H0[i, i+1] = vf * (k_vec[mod(i, k_space_size)+1] - 1im * k_vec[div(i, k_space_size)+1])
    end

    potential = kron(potential, I(2))
    H0 = kron(H0, I(2))
    sx = kron(I(k_space_size^2), sx2)
    ham = H0 + potential

    @assert ishermitian(ham) "julianic hamiltonian is not hermitian"

    vals, vecs = eigen(ham)
    conductivity = 0.0

    for (idx, E1) in enumerate(vals)
        for (jdx, E2) in enumerate(vals)
            if E1 != E2
                conductivity += abs2(vecs[:, idx]' * sx * vecs[:, jdx]) * (fermi_dirac(real(E1)) - fermi_dirac(real(E2))) / (E1 - E2) / (1im * eta * (E1 - E2))
            end
        end
    end

    conductivity *= factor
    
    # return kx_space, ky_space, potential
    return conductivity
end