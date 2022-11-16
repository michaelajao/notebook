using OrdinaryDiffEq, Plots

function ball!(du,u,p,t)
    du[1] = u[2]
    du[2] = 0.0
    du[3] = u[4]
    du[4] = -p[1]
end

ground_condition(u,t,integrator) = u[3]
ground_affect!(integrator) = integrator.u[4] = -integrator.p[2] * integrator.u[4]
ground_cb = ContinuousCallback(ground_condition, ground_affect!)

u0 = [0.0,2.0,50.0,0.0]
tspan = (0.0,50.0)
p = [9.807, 0.9]

prob = ODEProblem(ball!,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=ground_cb)
plot(sol, vars=(1,3), label = nothing, xlabel="x", ylabel="y")


results = "hidden"
stop_condition(u,t,integrator) = u[1] - 25.0
stop_cb = ContinuousCallback(stop_condition, terminate!)
cbs = CallbackSet(ground_cb, stop_cb)

tspan = (0.0, 1500.0)
prob = ODEProblem(ball!,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=cbs)


rectangle(xc, yc, w, h) = Shape(xc .+ [-w,w,w,-w]./2.0, yc .+ [-h,-h,h,h]./2.0)

begin
    plot(sol, vars=(1,3), label=nothing, lw = 3, c=:black)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
end


using Distributions

cor_dist = truncated(Normal(0.9, 0.02), 0.9-3*0.02, 1.0)
trajectories = 100000

prob_func(prob,i,repeat) = remake(prob, p = [p[1], rand(cor_dist)])
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
ensemblesol = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=trajectories, callback=cbs)

begin # plot
    plot(ensemblesol, vars = (1,3), lw=1,alpha=0.2, label=nothing)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing, c=:green)
    plot!(sol, vars=(1,3), label=nothing, lw = 3, c=:black, ls=:dash)
    xlims!(0.0,27.5)
end