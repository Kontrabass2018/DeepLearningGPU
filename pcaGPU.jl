using LinearAlgebra
using Statistics
function fit_transform_pca(X, outdim)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    U, S, V = JacobiGPU(Z,full=true);
    Matrix(U[:, sortperm(S, rev=true)[1:outdim]]') * Z
end
function fit_pca(X, outdim)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    U, S, V = JacobiGPU(Z,full=true);
    return Matrix(U[:, sortperm(S, rev=true)[1:outdim]]') 
end
function transform_pca(X, P)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    P * Z
end 