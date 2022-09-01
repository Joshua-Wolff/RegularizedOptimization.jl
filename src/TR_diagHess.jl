export TR_diagHess


```
New TR algorithm with diagonal hessian


Comment passer les options 

```

function TR_diagHess(
  f::AbstractNLPModel,
  h::H,
  options::ROSolverOptions;
  x0::AbstractVector = f.meta.x0,
  selected::UnitRange{T} = 1:length(f.meta.x0)) where {T <: Integer, H}

  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵ
  Δk = options.Δk
  Δmax = 1000 * options.Δk
  verbose = options.verbose
  maxIter = 1000000#options.maxIter
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  γ = options.γ
  has_bounds(f) ? (l_bound = f.meta.lvar ; u_bound = f.meta.uvar) : (l_bound = -Inf ; u_bound = Inf)

  # initialize parameters
  xk = copy(x0)
  hk = h(xk[selected])
  if hk ∈ (Inf,-Inf) # check if hk finite in x0
    error("hk not finite in x0")
  end
  xkn = similar(xk)
  s = zero(xk)
  # define ψ
  ψ = shifted(h, xk, max.(-Δk,l_bound .- xk), min.(Δk, u_bound .- xk), Δk, selected)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)

  # logs
  if verbose > 0
    @info @sprintf "%6s %8s %8s %7s %8s %7s %1s" "outer" "f(x)" "h(x)" "√ξ" "ρ" "Δ" "TR"
  end

  k = 0

  # compute objective and gradient
  fk = obj(f, xk)
  ∇fk = grad(f, xk)
  ∇fk⁻ = copy(∇fk)
  # compute initial diagonal hessian (for the moment we use the identity as initialization)
  Dk = DiagonalQN(ones(eltype(xk), length(xk)))
  #Dk = SpectralGradient(one(eltype(xk)),length(xk))

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime

  global ξ

  while !(optimal || tired)

    if isa(Dk.d, Number)
      if Dk.d == zero(eltype(Dk.d))
        Dk.d = 10*eps()
      elseif abs(Dk.d) < 10*eps()
        Dk.d = sign(Dk.d) * 10*eps()
      end
    else
      for i in 1:length(Dk.d) # non-zero diagonal coefficients
        if Dk.d[i] == zero(eltype(Dk.d))
          Dk.d[i] = 10*eps()
        elseif abs(Dk.d[i]) < 10*eps()
          Dk.d[i] = sign(Dk.d[i]) * 10*eps()
        end
      end 
    end

    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # linear model of the smooth part of the objective
    φ(d) = ∇fk' * d + 1/2 * d' * (Dk.d .* d)

    # define global model
    mk(d) = φ(d) + ψ(d)

    # compute step
    prox!(s, ψ, -∇fk ./ Dk.d, Dk)

    # compute ξ and ratio ρk
    xkn .= xk .+ s
    println(norm(s,2))
    fkn = obj(f, xkn)
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()
    if (ξ ≤ 0 || isnan(ξ))
      error("TR: failed to compute a step: ξ = $ξ")
    end
    if sqrt(ξ) < ϵ
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end
    ρk = Δobj / ξ

    # logs
    TR_stat = (η2 ≤ ρk < Inf) ? "↗" : (ρk < η1 ? "↘" : "=")
    if verbose > 0
      @info @sprintf "%6d %8.1e %8.1e %7.1e %8.1e %7.1e %1s" k fk hk sqrt(ξ) ρk ψ.Δ TR_stat
    end

    if η2 ≤ ρk < Inf # very successful step
      Δk = min(γ * Δk, Δmax)
      set_radius!(ψ, Δk)
    end

    if η1 ≤ ρk < Inf
      ∇fk = grad(f, xkn)
      push!(Dk, xkn - xk, ∇fk - ∇fk⁻)
      xk .= xkn
      fk = fkn
      hk = hkn
      set_bounds!(ψ, max.(-Δk, l_bound .- xk), min.(Δk, u_bound .- xk))
      shift!(ψ, xk)
      ∇fk⁻ .= ∇fk
    end

    if ρk < η1 || ρk == Inf
      Δk = Δk / 2
      set_bounds!(ψ, max.(-Δk, l_bound .- xk), min.(Δk, u_bound .- xk))
      set_radius!(ψ, Δk)
    end
    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      @info @sprintf "%6d %8.1e %8.1e %7.1e %7.1e" k fk hk sqrt(ξ) ψ.Δ
      @info "TR: terminating with √ξ = $(sqrt(ξ))"
    end
  end

  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end

  return GenericExecutionStats(
    status,
    f,
    solution = xk,
    objective = fk + hk,
    dual_feas = sqrt(ξ),
    iter = k,
    elapsed_time = elapsed_time,
    solver_specific = Dict(
      :Fhist => Fobj_hist[1:k],
      :Hhist => Hobj_hist[1:k],
      :NonSmooth => h,
    ),
  )

end