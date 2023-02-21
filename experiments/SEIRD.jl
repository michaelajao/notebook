using DifferentialEquations, Plots

function seird!(du, u, p, t)
    S, E, I, R, D = u
    λ, N, β, α, γ, μ, δ, ε = p
    
    dSdt = λ * N - β * S * I / N - δ * S + ε * R
    dEdt = β * S * I / N - α * E - δ * E
    dIdt = α * E - γ * I - μ * I - δ * I
    dRdt = γ * I - ε * R - δ * R
    dDdt = μ * I + μ * R
    
    du[1] = dSdt
    du[2] = dEdt
    du[3] = dIdt
    du[4] = dRdt
    du[5] = dDdt
end

# Initial conditions
u0 = [999999, 100, 10, 0, 0]

# Parameters
p = [0.05, 1000000, 0.7, 0.2, 0.1, 0.01, 0.01, 0.01]

# Time span
tspan = (0.0, 200.0)

# Solve the ODE system
prob = ODEProblem(seird!, u0, tspan, p)
sol = solve(prob)

# Plot the results
plot(sol, xlabel="Time", ylabel="Population",
    title="SEIRD Model Simulation", label=["S" "E" "I" "R" "D"])
