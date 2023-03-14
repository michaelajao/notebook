using DataFrames, Dates, Plots, CSV, DataFrameMacros, LaTeXStrings

function data_processing(country::String)
    data = CSV.read("data/confirmed_cases_global.csv", DataFrame)
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

start_date = Date(2021, 01, 01);
end_date = Date(2021, 03, 01);

ng_data = data_processing("Nigeria");

uk_data = data_processing("United Kingdom")

ng = filter(row -> start_date <= row.dates <= end_date, ng_data)
uk = filter(row -> start_date <= row.dates <= end_date, uk_data)


plot(ng[!, :dates],
    ng[!, :confirmed],
    title="Nigeria COVID-19 Confirmed Cases",
    xlab=L"time", ylab="infected daily",
    yformatter=y -> string(round(Int64, y ÷ 1_000)) * "K",
    label=false)



tstart = 0.0
tend = 10.0
sampling = 0.1

model_params = [1.5, 1.0, 3.0, 1.0]


model_params = [sqrt(0.3), sqrt(0.9), sqrt(0.19), sqrt(0.5), sqrt(0.01159)]

# take square root of all the numbers and then square them in the model
alpha_sr = sqrt(0.15)
gamma_sr = sqrt(0.00744)
delta_sr = sqrt(0.1)
sigma_sr = sqrt(0.9)

function seird!(du, u, p, t)
    # Unpack parameters
    (β, σ, γ, μ) = p

    # Unpack variables
    (S, E, I, R, D) = u

    # Calculate derivatives
    du[1] = -β * S * I
    du[2] = β * S * I - σ * E
    du[3] = σ * E - γ * I - μ * I
    du[4] = γ * I
    du[5] = μ * I
end

function predict_adjoint(param) # Our 1-layer neural network
    prob = ODEProblem(model, [1.0, 0.0], (tstart, tend), model_params)
    Array(concrete_solve(prob, Tsit5(), param[1:2], param[3:end], saveat=tstart:sampling:tend, abstol=1e-8, reltol=1e-6))
end