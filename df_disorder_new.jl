using LinearAlgebra
using MPI

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create output directory based on the rank

l_min, l_max = 10,40
num_lengths = 5

# Define L (equivalent of L array in Python)
L = range(l_min, stop=l_max, length=num_lengths)

k_grid = 50
n_config = 1 
u0 = 1
l0 = 1
# L = [10,20,30]
K = 20*2*pi*L.^-1
lamda = 0
epsilon = 1e-12


# sigmax = [[0,1], [1,0]]
# sigmay = [[0, -1im], [1im, 0]]
# sigmaz = [[1, 0], [0, -1]]
sigma0 = [1 0
          0 1]
sigmax = [0 1
          1 0]
sigmay = [0 -1im
          1im 0]
sigmaz = [1 0
          0 -1]
Ik = I(k_grid*k_grid)
g_avg = zeros(length(L), 1)
bg = zeros(length(L)-1, 1)

function htheta(x)
    if x >= 0
        result = 1;
    else
        result = 0;
    end
    return result
end

for lsize = 1:length(L)
    
    k = range(-K[lsize], K[lsize], k_grid)
    
    kq = 2*K[lsize]/k_grid
    eta = 2*kq
    # k_disc = hcat(k,k)
    kx = k'.*ones(k_grid)
    ky = ones(k_grid)'.*k
    kx = kx[:]
    ky = ky[:]
    k_disc = [kx' ky']
    # k_disc = Matrix(ComplexF64, k_disc)
    
    # println(size(kx))
    h1x = Diagonal(kx)#, 0, k_grid*k_grid, k_grid*k_grid ))
    h1y = Diagonal(ky)#, 0, k_grid*k_grid, k_grid*k_grid ))

    
    h1 = kron(h1x, sigmax) + kron(h1y, sigmay) + lamda*kron(h1x, sigma0)

    g = 0
    
        for iter = 1:n_config
             
            h3 = zeros(k_grid*k_grid, k_grid*k_grid)
            h2 = zeros(k_grid*k_grid, k_grid*k_grid)
            for r = 1:10*L[lsize] 
                x = L[lsize]*rand(1)
                y = L[lsize]*rand(1)
                for a = 1:length(k_disc[:,1])
                  
                    for j = 1:length(k_disc[:,1])
                        kx = k_disc[a,1] - k_disc[j,1]
                        ky = k_disc[a,2] - k_disc[j,2]
                        
                            hv = (u0/L[lsize]^2)*exp(-1im*kx*x[1] - 1im*ky*y[1])*exp(-(kx^2 + ky^2)*((l0)^2)/2)
                            h2[a, j] = hv
                       
                    end 
    
                end  
                h3 = h2 + h3
                
            end
            h4 = h1 + kron( h3, sigma0) 
            h5 = kron(Ik, sigmax)
            
            eigval, eigvec = eigen(h4)
            
            energies = zeros(100, 1)
    
            tem = 0
            
            for i = 2450:2550
                for j = 2450:2550
                    vec_i = eigvec[:,i]
                    vec_j = eigvec[:,j]
                    val_i = eigval[i]
                    val_j = eigval[j]
                    
                    x1 = (htheta(val_i) - htheta(val_j))/(val_i - val_j + epsilon)
                    
                    x2 = (abs(vec_i'*h5*vec_j))^2
                    tem  = tem + x1*x2/((val_i-val_j) + 1im*eta)
                  
                end    
            end 
            
            
            g = g + tem*2*1im*pi/L[lsize]^2

            if abs(imag(g)) < 1e-7
                g = real(g)
            end

            # println("config: ", iter, " g: ", g)
         
    
        end  
        
        g_avg[lsize] = g/n_config
    
    end     

    for i = 1:length(bg)
        bg[i] = (g_avg[i]+1 - g_avg[i])/(L[i+1]-L[i])*(L[i]/g_avg[i]);
    end   


dirname = joinpath("output_data", "results_version", "run$rank")
mkpath(dirname)  # This creates the directory if it doesn't exist


println("successfully completed")

# Check if the parameter file exists; if not, write it
params_file = joinpath("output_data", "results_version", "params.txt")
if !isfile(params_file)
    open(params_file, "w") do file
        text = """
        l_min, l_max = $l_min, $l_max
        u = $u
        l0 = $l0
        configurations = $n_config
        k_space_size = $k_grid
        """
        write(file, text)
        println("Parameter file written")
    end
end

fname = joinpath("output_data", "results_version", "length.npy")
if !isfile(fname)
    open(fname, "w") do f
        serialize(f, L)
    end
end

conductivities_str = string(bg)

function determine_next_filename(fname::String, folder::String, filetype::String)
    files = readdir(folder)
    i = 1
    while "$fname$i.$filetype" in files
        i += 1
    end
    return joinpath(folder, "$fname$i.$filetype")
end

    # File handling (create folder and determine the next filename)
fname = determine_next_filename("output", dirname, "txt")

# Write to the file
write_file(fname, conductivities_str)

# Finalize MPI
MPI.Finalize()