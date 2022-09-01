using Random
using LinearAlgebra
using ProximalOperators
using NLPModels, NLPModelsModifiers

include("/Users/joshuawolff/Documents/GERAD/src/TR_diagHess/RegularizedOptimization.jl/src/RegularizedOptimization.jl")
include("/Users/joshuawolff/Documents/GERAD/src/RegularizedProblems.jl/src/RegularizedProblems.jl")

using Printf

Random.seed!(1234)

function demo_solver(f, h; selected = 1:length(f.meta.x0))
  options = ROSolverOptions(ϵ = 1e-5, verbose = 10)
  @info " using TR to solve with" h
  #reset!(f)
  TR_out = TR_diagHess(f, h, options, selected = selected)
  TR_out
end

function demo_nnmf()
  model = nnmf_model(100,50,5)
  f = LSR1Model(model)
  λ = norm(grad(model, rand(model.meta.nvar)), Inf) / 10
  #res0 = demo_solver(f, NormL0(λ), selected = model.selected)
  res1 = demo_solver(f, NormL1(λ), selected = model.selected)
  return res1#, res1
end

function demo_bpdn_constr(compound = 1)
  model, sol = bpdn_constr_model(compound)
  f = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  res0 = demo_solver(f, NormL0(λ))
  res1 = demo_solver(f, NormL1(λ))
  return sol, res0, res1
end

function demo_bpdn(compound = 1)
  model, modelNLS, sol = bpdn_model(compound)
  f = LSR1Model(model)
  λ = norm(grad(model, zeros(model.meta.nvar)), Inf) / 10
  res0 = demo_solver(f, NormL0(λ))
  res1 = demo_solver(f, NormL1(λ))
  return sol, res0, res1
end

#bpdn_true, bpdn_res0, bpdn_res1 = demo_bpdn() # marche nickel
#bpdn_constr_true, bpdn_constr_res0 = demo_bpdn_constr() # marche nickel
nnmf_res1 = demo_nnmf()