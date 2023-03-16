using DifferentialEquations, Plots

# Define the SEIRD model
function seird!(du, u, p, t)
    # Unpack parameters
    (β, σ, γ, μ) = p

    # Unpack variables
    (S, E, I, R, D) = u

    # Calculate derivatives
    dS = -β * S * I
    dE = β * S * I - σ * E
    dI = σ * E - γ * I - μ * I
    dR = γ * I
    dD = μ * I

    # Update derivatives
    du[1] = dS
    du[2] = dE
    du[3] = dI
    du[4] = dR
    du[5] = dD
end

# Define initial conditions and parameters
u0 = [0.99, 0.01, 0.0, 0.0, 0.0]
p = (0.3, 0.2, 0.1, 0.01)

# Define time span and time step
tspan = (0.0, 200.0)
dt = 0.1

# Solve the differential equations
prob = ODEProblem(seird!, u0, tspan, p)
sol = solve(prob, Tsit5(), dt=dt)

# Plot the results
plot(sol, label=["S" "E" "I" "R" "D"], xlabel="Time", ylabel="Proportion", title="SEIRD Model")

savefig("data/SEIRD.png")
using Pkg
Pkg.instantiate()