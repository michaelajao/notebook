using DataFrames, Dates, Plots, CSV,  DataFrameMacros

# donwload the data:
# url_confirmed_cases= "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
# url_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
# url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

function data_processing(country)
    data = CSV.read("C:/Users/olarinoyem/Documents/GitHub/covid-19-forcasting-experiments/data/confirmed_cases_global.csv", DataFrame)
    rename!(data, 1 => "province", 2 => "country")
    countries = collect(data[:, 2])
    row = findfirst(countries .== country)
    data_row = data[row, :]
    country_data = [i for i in values(data_row[5:end])]
    
    date_strings =String.(names(data))[5:end]
    format = Dates.DateFormat("m/d/Y")
    dates = parse.(Date, date_strings, format) + Year(2000)

    df = DataFrame(country = country_data, dates = dates)    
    return df
end

ng_data = data_processing("Nigeria")

ng_filtered = @where(ng_data, country > 0)





using JuMP, Gurobi, Metaheuristics


#Define the parameters
n = length(confirmed_cases_timeseries) # number of time steps
m = length(small_world_network) # number of nodes in the network
k = length(vaccine_doses) # number of available vaccine doses

#Create the model
model = Model(solver=MetaheuristicsSolver())

#Define the decision variables
@variable(model, x[1:n, 1:m, 1:k], Bin) # binary variable indicating whether a dose of vaccine is allocated to node i at time t
@variable(model, y[1:n, 1:m], Bin) # binary variable indicating whether node i is in the infected state at time t

#Define the objective function
@objective(model, Min, sum(y[1:n, 1:m])) # minimize the number of infected nodes

#Define the constraints
#Ensure that only one dose of vaccine can be allocated to each node at each time step
@constraint(model, [t in 1:n, i in 1:m], sum(x[t,i,1:k]) == 1)

#Ensure that the number of allocated doses does not exceed the total number available
@constraint(model, [t in 1:n], sum(x[t,1:m,1:k]) <= k)

#Update the infected state of each node using the SEIR ODE and small world network dynamics
@constraint(model, [t in 2:n, i in 1:m], y[t,i] == y[t-1,i] + dt*(betay[t-1,i]sum(y[t-1,j] for j in neighbors[i]) - gammay[t-1,i] - muy[t-1,i]))

#Use the confirmed cases data to initialize the infected state of each node at the first time step
@constraint(model, [i in 1:m], y[1,i] == confirmed_cases_timeseries[1,i])

#Make the model NP-hard by adding a constraint that ensures the solution is unique
@constraint(model, [t in 1:n, i in 1:m, j in 1:k, l in 1:k], i != j || t != l || x[t,i,j] + x[l,i,j] <= 1)

#Solve the model
solve(model)

#Retrieve the optimal solution
optimal_allocation = getvalue(x)
infected_nodes = getvalue(y)

#Print the results
println("Optimal vaccine allocation:")
println(optimal_allocation)
println("Infected nodes at each time step:")
println(infected_nodes)




using JuMP, Gurobi

#Define the parameters
n = length(timeseries) # number of time steps
m = length(small_world_network) # number of nodes in the network
k = length(vaccine_doses) # number of available vaccine doses

#Create the model
model = Model(solver=GurobiSolver())

#Define the decision variables
@variable(model, x[1:n, 1:m, 1:k], Bin) # binary variable indicating whether a dose of vaccine is allocated to node i at time t
@variable(model, y[1:n, 1:m], Bin) # binary variable indicating whether node i is in the infected state at time t

#Define the objective function
@objective(model, Min, sum(y[1:n, 1:m])) # minimize the number of infected nodes

#Define the constraints
#Ensure that only one dose of vaccine can be allocated to each node at each time step
@constraint(model, [t in 1:n, i in 1:m], sum(x[t,i,1:k]) == 1)

#Ensure that the number of allocated doses does not exceed the total number available
@constraint(model, [t in 1:n], sum(x[t,1:m,1:k]) <= k)

#Update the infected state of each node using the SEIR ODE and small world network dynamics
@constraint(model, [t in 2:n, i in 1:m], y[t,i] == y[t-1,i] + dt*(betay[t-1,i]sum(y[t-1,j] for j in neighbors[i]) - gammay[t-1,i] - muy[t-1,i]))

#Solve the model
solve(model)

#Retrieve the optimal solution
optimal_allocation = getvalue(x)
infected_nodes = getvalue(y)

#Print the results
println("Optimal vaccine allocation:")
println(optimal_allocation)
println("Infected nodes at each time step:")
println(infected_nodes)




using DifferentialEquations
using Optim

#SEIRS ODE model
function SEIRS(dy, y, p, t)
    beta, sigma, gamma, mu = p
    S, E, I, R, D = y
    N = S + E + I + R + D
    dy[1] = -beta * S * I / N
    dy[2] = beta * S * I / N - sigma * E
    dy[3] = sigma * E - gamma * I - mu * I
    dy[4] = gamma * I
    dy[5] = mu * I
    end

    #optimization function to find the optimal values of beta, sigma, gamma, and mu
function optimize_SEIRS(params, y0, t, data)
    # run the ODE model
    result = solve(ODEProblem(SEIRS, y0, t, params), Tsit5(), saveat=t)
    # calculate the error between the model results and the data
    error = sum((result - data).^2)
    return error
end

#initial values for the ODE model
y0 = [S0, E0, I0, R0, D0]

#time points for the ODE model
t = range(0, T, step=T/num_points)

#data for confirmed cases in multiple regions
data = [region1_cases, region2_cases, ...]

#bounds for beta, sigma, gamma, and mu
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]

#initial values for beta, sigma, gamma, and mu
params0 = [beta0, sigma0, gamma0, mu0]

#minimize the error between the ODE model and the data using metaheuristics
result = optimize(optimize_SEIRS, bounds, y0=y0, t=t, data=data, method=:metaheuristics)

#print the optimal values for beta, sigma, gamma, and mu
println(result.minimizer)



using JuMP
using Gurobi
using Ipopt

function optimize_vaccine_allocation(SEIR_dynamics::Array,vaccination_rates::Array, time_periods::Int, regions::Int)
    # Initialize JuMP model
    model = Model(with_optimizer(Ipopt.Optimizer))

    # Define decision variables for vaccine allocation
    @variable(model, vaccine_allocation[1:time_periods, 1:regions] >= 0)

    # Define objective function to minimize the number of infected individuals in each region
    @objective(model, Min, sum(vaccine_allocation .* SEIR_dynamics[:, :, 2]))

    # Constraint to ensure that the total vaccination rate does not exceed the available vaccines
    @constraint(model, sum(vaccine_allocation) <= vaccination_rates[time_periods])

    # Constraint to ensure that the vaccination rate in each region does not exceed the maximum capacity
    @constraint(model, vaccine_allocation[time_periods, :] <= vaccination_rates[time_periods, :])

    # Solve the optimization problem
    optimize!(model)

    # Return the optimal vaccine allocation
    return value.(vaccine_allocation)
end

Define SEIR dynamics for each region
SEIR_dynamics = [rand(0:10, 3, 3) for i in 1:regions]

Define vaccination rates for each time period and region
vaccination_rates = rand(0:100, time_periods, regions)

Solve the optimization problem
optimal_vaccine_allocation = optimize_vaccine_allocation(SEIR_dynamics, vaccination_rates, time_periods, regions)

Print the optimal vaccine allocation
println("Optimal vaccine allocation: ", optimal_vaccine_allocation)





using Turing
using DataFrames
using Distributions

#Load data from John Hopkins COVID-19 Github repo
data = DataFrame(CSV.File("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"))

Define SEIR ODE model
@model SEIR(data::DataFrame) = begin
# Define parameters
β ~ Beta(1, 1)
γ ~ Beta(1, 1)
λ ~ Beta(1, 1)
σ ~ Beta(1, 1)
δ ~ Beta(1, 1)
ε ~ Beta(1, 1)
η ~ Beta(1, 1)
θ ~ Beta(1, 1)
ι ~ Beta(1, 1)
κ ~ Beta(1, 1)

#Copy code
# Define ODE system
S = Vector{Float64}(undef, length(data))
E = Vector{Float64}(undef, length(data))
I = Vector{Float64}(undef, length(data))
R = Vector{Float64}(undef, length(data))
D = Vector{Float64}(undef, length(data))
S[1] = data[1]
E[1] = data[2]
I[1] = data[3]
R[1] = data[4]
D[1] = data[5]
for i in 2:length(data)
    S[i] = S[i - 1] - β * S[i - 1] * I[i - 1]
    E[i] = E[i - 1] + β * S[i - 1] * I[i - 1] - λ * E[i - 1]
    I[i] = I[i - 1] + λ * E[i - 1] - γ * I[i - 1] - δ * I[i - 1]
    R[i] = R[i - 1] + γ * I[i - 1]
    D[i] = D[i - 1] + δ * I[i - 1]
end

# Define likelihood
for i in 1:length(data)
    I[i] ~ Poisson(exp(S[i] * ε + E[i] * η + I[i] * θ + R[i] * ι + D[i] * κ))
end
end

#Sample from the posterior distribution of the parameters
chain = sample(SEIR(data), NUTS(0.65), 3000, thin = 10)

#Print the estimated parameters
println("Estimated parameters: ", mean(chain))


    # First, we need to include the necessary packages
    using DifferentialEquations
    using Plots
    using StatsFuns
    
    # Next, we need to obtain the data for the confirmed cases from the John Hopkins GitHub repository
    # We will assume that the data has been downloaded and is stored in a file called "cases.csv"
    cases = CSV.read("cases.csv")
    
    # We can then use the DifferentialEquations package to set up the SEIR ODE model
    function seir_model(du, u, p, t)
      S = u[1]
      E = u[2]
      I = u[3]
      R = u[4]
    
      du[1] = -p[1] * S * I
      du[2] = p[1] * S * I - p[2] * E
      du[3] = p[2] * E - p[3] * I
      du[4] = p[3] * I
    end
    
    # We also need to specify the initial conditions for the model
    u0 = [S0, E0, I0, R0]
    
    # We can then define a likelihood function and a prior distribution for the model parameters
    likelihood(p) = pdf(Normal(mean(cases), std(cases)), seir_model(u0, p, t))
    prior = [pdf(Uniform(0, 1), p[1]), pdf(Uniform(0, 1), p[2]), pdf(Uniform(0, 1), p[3])]
    
    # We can use the Metropolis-Hastings algorithm to sample from the posterior distribution of the model parameters
    posterior = metropolis_hastings(likelihood, prior)
    
    # Finally, we can use the Plots package to visualize the results of the parameter estimation
    plot(posterior)