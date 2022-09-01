using Plots

# bpdn
x = 1:length(bpdn_true)
y = zeros((length(bpdn_true),2))
y[:,1] = bpdn_true
y[:,2] = bpdn_res0.solution

#plot(x,y)

# bpdn_constr
x = 1:length(bpdn_constr_true)
y = zeros((length(bpdn_constr_true),2))
y[:,1] = bpdn_constr_true
y[:,2] = bpdn_constr_res0.solution

plot(x,y)