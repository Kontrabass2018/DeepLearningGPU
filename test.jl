include("functions.jl");
TCGA_data, labels, samples, genes, biotypes = fetch_data("TCGA_TPM_lab.h5")
TCGA_data_cds = TCGA_data[:,biotypes .== "protein_coding"]

# outfile = h5open("TCGA_19962_TPM_lab.h5", "w")
# outfile["data"] = TCGA_data_cds
# outfile["labels"] = labels 
# outfile["samples"] = samples 
# outfile["genes"] = genes[biotypes .== "protein_coding"]
# outfile["biotypes"] = biotypes[biotypes .== "protein_coding"]
# close(outfile)

TCGA_data, labels, samples, genes, biotypes = fetch_data("TCGA_19962_TPM_lab.h5")

TCGA_data
x_means =  mean(TCGA_data', dims =2 )
Z = TCGA_data' .- x_means




function JacobiPermute(p,q, X)
    Hpp, Hqq, Hpq = (X[p,p], X[q,q], X[p,q])
    cot2phik = (Hqq - Hpp) / (2 * Hpq)
    tanPhik = cot2phik / abs(cot2phik) /(abs(cot2phik) + sqrt(1 + cot2phik ^ 2))
    find_sin(TAN) = TAN / sqrt(1 + TAN ^ 2) 
    find_cos(TAN) = 1 / sqrt(1 + TAN ^ 2)
    sinPhik = find_sin(tanPhik)
    cosPhik = find_cos(tanPhik)
    #J = zeros(n,n);[J[i,i] = 1 for i in 1:n];
    Jtr = [cosPhik sinPhik; -sinPhik cosPhik]
    
    return Jtr
end 
l2norm(Vec) = sum(abs2, Vec)
function criterion(Xp, Xq;eps=1e-7)
    return abs(Xp' * Xq) > (eps * l2norm(Xp) * l2norm(Xq))
end 
X = Matrix(Z) |> gpu 
n,m = size(TCGA_data)
n = min(n,m)

Vx = zeros(n,n) ;[Vx[i,i] = 1 for i in 1:n]; Vx = Vx |>gpu
max_iter = 10 
converged = false; iter = 1
h1,h2, h3, h4 = vec(X[[1,2],[1,2]])
while !converged && iter < max_iter
    converged = true 
    println("$iter")
    for a in 1:n 
        println("$a")
        for b in 1:n
            if b > a && criterion(X[:,a], X[:,b], eps=1e-7)
                Jtr = JacobiPermute(a, b, X)
                # find jacobi angle 
                # apply transfo 
                # update params 
                Vx[[a,b],[a,b]] .= Vx[[a,b],[a,b]] * Jtr 
                X[[a,b],[a,b]] .= X[[a,b],[a,b]] * Jtr
                converged = false 
            end  
        end 
    end
    iter += 1 
end 