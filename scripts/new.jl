using DifferentialEquations, Plots





function SEIRS(du, u, p, t)
    S,E,I,R = u
    β, ω, γ_r, γ_i, ρ, σ = p

    du[1] = (-β * S * I) - ω*S + γ_r * R
    du[2] = (β * S * I)- σ*E - ω*E
    du[3] = (σ * ρ * E) - (γ_i * I) - (ω * I)
    du[4] = (γ_i * I) - (γ_r * R) + ω * (S+E+I)
end

p = [0.1, 0.1, 0.1,0.1, 0.1, 0.1 ]

init = [0.99,0.21,0.2,0.0]
tspan = (0.0, 100.0)

prob = ODEProblem(SEIRS, init, tspan, p)

sol = solve(prob, Tsit5())

plot(sol)


url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

using CSV, DataFrames, Dates

download(url, "covid_data.csv")

data = CSV.read("covid_data.csv", DataFrame)
rename!(data, 1 => "province", 2 => "country")
countries = collect(data[:, 2])

unique_countries = unique(countries)
# u_countries = [startswith(country, "U") for country in countries]
# data[u_countries, :]

US_row = findfirst(countries .== "US")
us_data_row = data[US_row, :]

us_data = [i for i in values(us_data_row[5:end])]

plot(us_data)


date_strings =String.(names(data))[5:end]

format = Dates.DateFormat("m/d/Y")
dates = parse.(Date, date_strings, format) + Year(2000)

plot(dates, us_data)


function data_processing(country)
    data = CSV.read("covid_data.csv", DataFrame)
    rename!(data, 1 => "province", 2 => "country")
    countries = collect(data[:, 2])
    row = findfirst(countries .== country)
    data_row = data[row, :]
    country_data = [i for i in values(data_row[5:end])]

    date_strings =String.(names(data))[5:end]

    format = Dates.DateFormat("m/d/Y")
    dates = parse.(Date, date_strings, format) + Year(2000)

    return country_data
end


ng_data = data_processing("Nigeria")

