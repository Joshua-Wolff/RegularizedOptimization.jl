#module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf
include("/Users/joshuawolff/Documents/GERAD/src/LinearOperators.jl/src/LinearOperators.jl")

# external dependencies
using Arpack, ProximalOperators

# dependencies from us
using NLPModels, NLPModelsModifiers, SolverCore
include("/Users/joshuawolff/Documents/GERAD/src/ShiftedProximalOperators.jl/src/ShiftedProximalOperators.jl")

include("utils.jl")
include("input_struct.jl")
include("PG_alg.jl")
include("Fista_alg.jl")
include("splitting.jl")
include("TR_alg.jl")
include("R2_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")
include("TR_diagHess.jl")

#end  # module RegularizedOptimization
