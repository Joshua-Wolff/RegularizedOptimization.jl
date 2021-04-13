module TRNC
using LinearAlgebra, Printf, Roots, Random

export TRNCPATH
TRNCPATH = dirname(@__DIR__)

include("descentopts.jl")
include("proxgrad.jl")
include("fista.jl")
include("splitting.jl")
include("linesearch.jl")
include("TR_alg.jl")
include("QR_alg.jl")
# include("proxGD.jl")


end  # module TRNC
