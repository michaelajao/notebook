using DataFrames, Dates, Plots, CSV, DataFramesMeta, MLJ
using GaussianProcesses: GaussianProcessesRegressor

function data_processing(country::String)
    data = CSV.read("C:/Users/olarinoyem/Documents/GitHub/covid-19-forcasting-experiments/data/confirmed_cases_global.csv", DataFrame)
    rename!(data, 1 => "province", 2 => "country")
    countries = collect(data[:, 2])
    row = findfirst(countries .== country)
    data_row = data[row, :]
    country_data = [i for i in values(data_row[5:end])]
    
    date_strings =String.(names(data))[5:end]
    format = Dates.DateFormat("m/d/Y")
    dates = parse.(Date, date_strings, format) + Year(2000)

    df = DataFrame(confirmed = country_data, dates = dates)    
    return df
end



ng_data = data_processing("Nigeria")

ng_filtered = @subset(ng_data, :confirmed .> 0)

start_date = Date(2020,03,01)
end_date = Date(2020,12,01)

# range_data = ng_filtered[2,start_date:end_date]

selected_rows = filter(row -> start_date <= row.dates <= end_date, ng_filtered)

using Plots, StatsPlots, LaTeXStrings
@df selected_rows plot(:dates,
    :confirmed,
    xlab=L"time", ylab="infected daily",
    yformatter=y -> string(round(Int64, y ÷ 1_000)) * "K",
    label=false)



us_data =  data_processing("US")

selected_rows = filter(row -> start_date <= row.dates <= end_date, us_data)

@df selected_rows plot(:dates,
    :confirmed,
    xlab=L"time", ylab="infected daily",
    yformatter=y -> string(round(Int64, y ÷ 1_000)) * "K",
    label=false)


# x = selected_rows[!, :dates]
# y = selected_rows[!, :confirmed]

# gpr = GP()


# # Plot the original time series data
# plot(x, y, label="Original Data")

# # Plot the normalized time series data
# plot!(normalized_time_series, label="Normalized Data")




# using GaussianProcesses, Plots, Random

# function normalize_and_plot(y, new_times)
#     # Fit a Gaussian process regressor to the time series data
#     gpr = gaussian_process(y, MeanZero(), SE(1.0,0.0))

#     # Make predictions about future values of the time series
#     mean, cov = predict(gpr, new_times)

#     # Plot the original data and the predicted values
#     plot(new_times, y, label="Original Data")
#     plot!(new_times, mean, label="Predicted Values")
# end


# y = selected_rows[!, :confirmed]

# new_times = selected_rows[!, :dates]

# normalize_and_plot(y, new_times)

# #Simulate the data
# Random.seed!(203617)

# using DifferentialEquations
# using Plots

# #Initialize parameters
# beta = 0.2
# gamma = 0.1
# delta = 0.01
# sigma = 0.05

# #Initialize population sizes
# s0 = 0.99
# e0 = 0.11
# i0 = 0.01
# r0 = 0

# #Define SEIR model
# function SEIR(du, u, p, t)
# S, E, I, R = u
# du[1] = -beta*S*I
# du[2] = beta*S*I - sigma*E
# du[3] = sigma*E - gamma*I
# du[4] = gamma*I
# end

# #Define initial conditions
# u0 = [s0, e0, i0, r0]

# #Define time range for simulation
# tspan = (0.0, 100.0)

# #Solve SEIR model
# prob = ODEProblem(SEIR, u0, tspan)
# sol = solve(prob, Tsit5())

# #Extract data from solution
# s = sol[1,:]
# e = sol[2,:]
# i = sol[3,:]
# r = sol[4,:]

# #Plot results
# plot(sol.t, s, label="Susceptible")
# plot!(sol.t, e, label="Exposed")
# plot!(sol.t, i, label="Infected")
# plot!(sol.t, r, label="Recovered")

# #Calculate demand for vaccine
# vaccine_demand = i .* s

# #Optimize vaccine allocation
# vaccine_allocation = optimize_vaccine_allocation(vaccine_demand)

# println("Optimized vaccine allocation: ", vaccine_allocation)

# # Import necessary packages
# using DifferentialEquations, Optim, Plots

# # Define function to solve the SEIR model
# function seir_model(du, u, p, t)
#   # Unpack parameters
#   β, γ, δ, σ = p
  
#   # Unpack state variables
#   S, E, I, R = u
  
#   # Define the differential equations for the SEIR model
#   dS = -β * S * I
#   dE = β * S * I - γ * E - σ * E
#   dI = γ * E - δ * I
#   dR = δ * I
  
#   # Set the values of the derivatives
#   du .= [dS, dE, dI, dR]
# end

# # Define function to optimize the SEIR model parameters
# function optimize_seir_model(p, t, y)
#   # Solve the SEIR model using the given parameters
#   u0 = [y[1], y[2], y[3], 0]
#   prob = ODEProblem(seir_model, u0, t, p)
#   sol = solve(prob,Tsit5())
  
#   # Compute the sum of squared errors between the model solution and the data
#   sse = sum((sol.u .- y).^2)
  
#   # Return the sum of squared errors
#   return sse
# end

# # Define initial parameter values
# p0 = [0.5, 0.2, 0.1, 0.1]

# # Define time points
# t = range(0, stop=365, length=365)

# # Load COVID-19 timeseries data
# # (

# # load required libraries
# using DifferentialEquations, Plots

# # define SEIR model
# function seir(du, u, p, t)
#   S, E, I, R = u
#   β, γ = p
  
#   du[1] = -β * S * I
#   du[2] = β * S * I - γ * E
#   du[3] = γ * E - γ * I
#   du[4] = γ * I
# end

# # define initial values and parameters
# u0 = [S₀, E₀, I₀, R₀]
# p = [β, γ]

# # define time span
# tspan = (0.0, 365.0)

# # solve ODE using the `seir` model
# prob = ODEProblem(seir, u0, tspan, p)
# sol = solve(prob, Tsit5())

# # extract the demand data for vaccine allocation
# demand = sol.u[3, :]

# # plot the demand data over time
# plot(sol.t, demand, xlabel="Time (days)", ylabel="Demand for vaccine")



using Graphs

# Function to generate a small world network
function small_world_network(n, k, beta)
    g = Graph(n)
    for i in 1:n
        for j in (i+1):(i+k)
            add_edge!(g, i, j % n + 1)
        end
    end

    # rewiring edges with probability beta
    for i in 1:n
        for j in neighbors(g, i)
            if rand() < beta
                neighbor = rand(1:n)
                if !has_edge(g, i, neighbor)
                    add_edge!(g, i, neighbor)
                    rem_edge!(g, i, j)
                end
            end
        end
    end
    return g
end

# Generate a small world network with n = 1000, k = 10, beta = 0.1
g = small_world_network(1000, 10, 0.1)


using Graphs
using Plots

# Function to generate a small world network
function small_world_network(n, k, beta)
    g = Graph(n)
    for i in 1:n
        for j in (i+1):(i+k)
            add_edge!(g, i, j % n + 1)
        end
    end

    # rewiring edges with probability beta
    for i in 1:n
        for j in neighbors(g, i)
            if rand() < beta
                neighbor = rand(1:n)
                if !has_edge(g, i, neighbor)
                    add_edge!(g, i, neighbor)
                    rem_edge!(g, i, j)
                end
            end
        end
    end
    return g
end

# Generate a small world network with n = 1000, k = 10, beta = 0.1
g = small_world_network(1000, 10, 0.1)

# Plot the graph
plot(g, nodelabel = 1:nv(g), layout = (x,y) -> [(cos(2 * pi * i / nv(g)), sin(2 * pi * i / nv(g))) for i in 1:nv(g)])
