using DifferentialEquations

function seird_nbd!(du, u, p, t)
    S, E, I, R, D = u
    λ, μ, β, α, γ, ε, δ, ν = p
    N = S + E + I + R + D
    
    du[1] = λ*N - β*S*I/N - δ*S + μ*N - ν*S
    du[2] = β*S*I/N - (α + δ)*E
    du[3] = α*E - (γ + μ + δ)*I
    du[4] = γ*I - (ε + δ)*R + ν*S
    du[5] = ε*R - (μ + δ)*D
end

# initial conditions
u0 = [1000000, 10, 0, 0, 0] # 999 susceptible, 1 exposed, 0 infectious, 0 recovered, 0 dead

# model parameters
p = [0.02, 0.01, 1.5, 0.2, 0.1, 0.05, 0.01, 0.0] # λ, μ, β, α, γ, ε, δ, ν

# time span
tspan = (0.0, 365.0)

# solve the differential equations
prob = ODEProblem(seird_nbd!, u0, tspan, p)
sol = solve(prob)

# plot the results
# using Plots
# plot(sol.t, sol[1,:], label="S")
# plot!(sol.t, sol[2,:], label="E")
# plot!(sol.t, sol[3,:], label="I")
# plot!(sol.t, sol[4,:], label="R")
# plot!(sol.t, sol[5,:], label="D")
# xlabel!("Time")
# ylabel!("Population")
# title!("SEIRD with Natural Birth and Death")


# Plot the results
plot(sol, xlabel="Time", ylabel="Population",
    title="SEIRD Model Simulation", label=["S" "E" "I" "R" "D"])