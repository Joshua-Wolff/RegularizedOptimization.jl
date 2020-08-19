export hardproxl1B2

function hardproxl1B2(s⁻, νλ; options)
    # %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
    # %constraint for a scalar variable
    xk = options.xk
    Δ = options.Δ
    
    ProjB(y) = min.(max.(y, -gk.-λν),-gk.+λν)
    froot(η) = η - norm(ProjB((-xk).*(η/Δ), gk))
    
    
    # %do the 2 norm projection
    y1 = ProjB(-xk) #start with eta = tau
    if (norm(y1)<= Δ)
        y = y1  # easy case
        str = "y in tau"
    else
        # η = fzero(froot, τ)
        η = fzero(froot, 1e-10, Inf)
        y = ProjB((-xk).*(η/Δ))
        str = "y root"
    end
    
    if(norm(y)<=Δ)
        s = y
        str2 = "within tau"
    else
        s = Δ.*y./norm(y)
        str2 = "out tau"
    end
    
    # @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
    return s 
end



# function hardproxB2(q, x, ν, λ, τ)
# function hardproxl1B2(Fcn, s⁻, ProjB, options)
# # %HARDPROXB2 computes the prox of the sum of shifted 1-norm and L2
# # %constraint for a scalar variable
# λ = options.λ
# ν = 1.0/options.β
# Bk = options.Bk
# xk = options.xk
# gk = options.∇fk
# Δ = options.Δ


# froot(η) = η - norm(ProjB((-xk).*(η/Δ), gk, ν))


# # %do the 2 norm projection
# y1 = ProjB(-xk, gk, ν) #start with eta = tau
# if (norm(y1)<= Δ)
#     y = y1  # easy case
#     str = "y in tau"
# else
#     # η = fzero(froot, τ)
#     η = fzero(froot, 1e-10, Inf)
#     y = ProjB((-xk).*(η/Δ), gk, ν)
#     str = "y root"
# end

# if(norm(y)<=Δ)
#     s = y
#     str2 = "within tau"
# else
#     s = Δ.*y./norm(y)
#     str2 = "out tau"
# end
# f = sum(Fcn(s, gk, xk, ν)) #because you need it to be component-wise

# # @printf("Y-meth: %s    s-meth: %s    s: %1.4e   y:%1.4e\n", str, str2, s[1], y[1]);
# return s,s⁻, f,1 #funEvals=1 here


# end
