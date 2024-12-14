using Pkg 
Pkg.activate(".")
# Pkg.add("cuDNN")
# Pkg.add("CUDA")
# Pkg.add("HDF5")
# Pkg.add("CSV")
using cuDNN
using CUDA
#data related
using HDF5
using CSV 
using Random

using Flux
using DataFrames
using CairoMakie

function load_tcga_data(infilename; shfl = true)
    infile = h5open(infilename)
    TCGA_data = infile["data"][:,:]
    labs = string.(infile["labels"][:])
    samples = string.(infile["samples"][:])
    genes = string.(infile["genes"][:]) 
    biotypes = string.(infile["biotypes"][:])
    close(infile)
    ids = collect(1:size(labs)[1])
    shfl && (ids = shuffle(ids))
    return TCGA_data[ids,:], labs[ids], samples[ids], genes, biotypes
end 

function fetch_data(filename; shfl = true)
    if !(filename in readdir("."))
        # Define the URL`
        tcga_data_url = "https://bioinfo.iric.ca/~sauves/VARIA/$filename"

        # Escape the URL to handle special characters
        escaped_url = Base.shell_escape(tcga_data_url)

        # Construct and execute the wget command
        command = `wget $escaped_url`
        run(command) 
    end 
    load_tcga_data(filename; shfl = shfl)

end 


function split_train_test(X::Matrix; nfolds = 5)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size  = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        train_x = X[tr_ids,:]
        # train_y = targets[tr_ids, :]
        test_x = X[tst_ids, :]
        # test_y = targets[tst_ids, :]
        folds[i] = Dict("foldn" => i, "train_x"=> train_x, "train_ids"=>tr_ids,"test_x"=> test_x, "test_ids" =>tst_ids)
    end
    return folds  
end