using MKL
using MPI
using LinearAlgebra
using Random
using Dates
using Printf
using Serialization  # For file handling
using DelimitedFiles  # For saving parameter files
using Distributions

# Function to determine the next filename (similar to determine_next_filename in Python)
function determine_next_filename(fname::String, folder::String, filetype::String)
    files = readdir(folder)
    i = 1
    while "$fname$i.$filetype" in files
        i += 1
    end
    return joinpath(folder, "$fname$i.$filetype")
end

# Function to write to a file (similar to write_file in Python)
function write_file(fname::String, data::String)
    open(fname, "w") do f
        write(f, data)
    end
end

# Main computation
export conductivity

l_min, l_max = 10,40
num_lengths = 15

vf = 1
h_cut = 1
u = 10
l0 = l_min / 30
N_i = 10
L = 10
# l0 = L/30
eta_factor = 1
T = 0
ef = 0

configurations = 1
k_space_size = 21

interaction_distance = 3

sx2 = [0 1; 1 0]
sy2 = 1im * [0 -1; 1 0]
sz2 = I(2)
sz2[end, end] = -1

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

    vals, vecs = eigen(ham)

    vals = vals[1:100]
    vecs = vecs[:, 1:100]
    
    conductivity = 0.0

    for (idx, E1) in enumerate(vals)
        for (jdx, E2) in enumerate(vals)
            if E1 != E2
                conductivity += abs2(vecs[:, idx]' * sx * vecs[:, jdx]) * (fermi_dirac(E1) - fermi_dirac(E2)) / (E1 - E2) / (1im * eta + (E1 - E2))
            end
        end
    end

    conductivity *= factor
    
    # return kx_space, ky_space, potential
    return conductivity
end

function main(L, rank)
    start_time = Dates.now()

    conductivities = [conductivity(l, eta_factor, rand(Uniform(-l/2, l/2), 2, N_i*floor(Int, l)), u, l0) for l in L]

    end_time = Dates.now()
    execution_time = round(Dates.value(end_time - start_time) / 1e9, digits=3)  # in seconds

    println("Conductivities computed in $execution_time seconds")

    # Convert the result to a string (in Julia, we can use `string` for conversion)
    conductivities_str = string(conductivities)

    # File handling (create folder and determine the next filename)
    dirname = joinpath("output_data", "results_version", "run$rank")
    mkpath(dirname)  # Create the directory if it doesn't exist
    fname = determine_next_filename("output", dirname, "txt")

    # Write to the file
    write_file(fname, conductivities_str)

    println("Conductivities stored")
    
end



# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create output directory based on the rank
dirname = joinpath("output_data", "results_version", "run$rank")
mkpath(dirname)  # This creates the directory if it doesn't exist

# Define L (equivalent of L array in Python)
L = range(l_min, stop=l_max, length=num_lengths)

# Call the main function (assuming you've defined `main` in Julia)

for _ in 1:configurations
    main(L, rank)
end

# Define the filename for saving the length array
fname = joinpath("output_data", "results_version", "length.npy")

# Check if the file exists before saving the array L[1] (just like np.save in Python)
if !isfile(fname)
    open(fname, "w") do f
        serialize(f, L[1])
    end
end

# Check if the parameter file exists; if not, write it
params_file = joinpath("output_data", "results_version", "params.txt")
if !isfile(params_file)
    open(params_file, "w") do file
        text = """
        l_min, l_max = $l_min, $l_max
        eta = $eta_factor * $vf * 2 * π / L
        vf = $vf
        h_cut = $h_cut
        u = $u
        l0 = $l0
        N_i = $N_i
        T = $T
        ef = $ef
        configurations = $configurations
        k_space_size = $k_space_size
        scattering potential = $pot_function
        """
        write(file, text)
        println("Parameter file written")
    end
end

# Finalize MPI
MPI.Finalize()
