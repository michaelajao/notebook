using Base.Threads
Threads.nthreads()


using DataFrames, Dates, Plots, CSV, DataFramesMeta, MLJ, StatsPlots, LaTeXStrings, DifferentialEquations
using Turing
using LazyArrays
using Random: seed!

function data_processing(country::String)
    data = CSV.read("../data/confirmed_cases_global.csv", DataFrame)
    rename!(data, 1 => "province", 2 => "country")
    countries = collect(data[:, 2])
    row = findfirst(countries .== country)
    data_row = data[row, :]
    country_data = [i for i in values(data_row[5:end])]

    date_strings = String.(names(data))[5:end]
    format = Dates.DateFormat("m/d/Y")
    dates = parse.(Date, date_strings, format) + Year(2000)

    df = DataFrame(confirmed=country_data, dates=dates)
    return df
end

start_date = Date(2020, 03, 01);
end_date = Date(2020, 12, 01);

ng_data = data_processing("Nigeria");

ng = filter(row -> start_date <= row.dates <= end_date, ng_data)

@df ng plot(:dates,
    :confirmed,
    xlab=L"time", ylab="infected daily",
    yformatter=y -> string(round(Int64, y ÷ 1_000)) * "K",
    label=false)


function sir_ode!(du, u, p, t)
    (S, I, R) = u
    (β, γ) = p
    N = S + I + R
    infection = β * I * S / N
    recovery = γ * I
    @inbounds begin
        du[1] = -infection # Susceptible
        du[2] = infection - recovery # Infected
        du[3] = recovery # Recovered
    end
    nothing
end;


i₀ = first(ng[:, :confirmed])
N = 2000000

u = [N - i₀, i₀, 0.0]
p = [0.5, 0.05]
prob = ODEProblem(sir_ode!, u, (1.0, 100.0), p)
sol_ode = solve(prob)
plot(sol_ode, label=[L"S" L"I" L"R"],
    lw=3,
    xlabel=L"t",
    ylabel=L"N",
    yformatter=y -> string(round(Int64, y ÷ 1_000_000)) * "mi",
    title="SIR Model for 100 days, β = $(p[1]), γ = $(p[2])")

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end




seed!(123)

@model function bayes_sir(infected, i₀, r₀, N)
    #calculate number of timepoints
    l = length(infected)

    #priors
    β ~ TruncatedNormal(2, 1, 1e-4, 10)     # using 10 because numerical issues arose
    γ ~ TruncatedNormal(0.4, 0.5, 1e-4, 10) # using 10 because numerical issues arose
    ϕ⁻ ~ truncated(Exponential(5); lower=0, upper=1e5)
    ϕ = 1.0 / ϕ⁻

    #ODE Stuff
    I = i₀
    u0 = [N - I, I, r₀] # S,I,R
    p = [β, γ]
    tspan = (1.0, float(l))
    prob = ODEProblem(sir_ode!,
        u0,
        tspan,
        p)
    sol = solve(prob,
        Tsit5(), # similar to Dormand-Prince RK45 in Stan but 20% faster
        saveat=1.0)
    solᵢ = Array(sol)[2, :] # New Infected
    solᵢ = max.(1e-4, solᵢ) # numerical issues arose

    #likelihood
    infected ~ arraydist(LazyArray(@~ NegativeBinomial2.(solᵢ, ϕ)))
end;



infected = ng[:, :confirmed]
r₀ = 1
model_sir = bayes_sir(infected, i₀, r₀, N)
chain_sir = sample(model_sir, NUTS(), 1_000)
summarystats(chain_sir[[:β, :γ]])

plot(chain_sir)