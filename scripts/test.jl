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


x = selected_rows[!, :dates]
y = selected_rows[!, :confirmed]

gpr = GP()


using Plots

# Plot the original time series data
plot(x, y, label="Original Data")

# Plot the normalized time series data
plot!(normalized_time_series, label="Normalized Data")






