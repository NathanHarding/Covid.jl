module GSA_write

export GSAwrite

using CSV
using DataFrames
using Dates
using Tables
using YAML
using Logging


"Filter large file, retaining rows with colname == value."
function filter_result(infile::String, outfile::String, colname::Symbol, value::String)
    result = Dict{Symbol, String}[]  # Each element is a row of the result. All types are strings...saves parsing
    for row in CSV.Rows(infile; delim=',', use_mmap=true, type=String)
        row[colname] != value && continue  # Row is excluded
        d = Dict{Symbol, String}()
        for cname in propertynames(row)
            d[cname] = row[cname]
        end
        push!(result, d)
    end
    result = DataFrame(result)
    CSV.write(outfile, result; delim=',')
end

function get_params(configfile::String)
    d = YAML.load_file(configfile)
    parameters = [
    d["distancing_policy"][Date("2020-01-01")]["household"],
    d["distancing_policy"][Date("2020-01-01")]["school"],
    d["distancing_policy"][Date("2020-01-01")]["workplace"],
    d["distancing_policy"][Date("2020-01-01")]["community"],
    d["distancing_policy"][Date("2020-01-01")]["social"],
    d["testing_policy"][Date("2020-01-01")]["IS"],
    d["testing_policy"][Date("2020-01-01")]["W"],
    d["tracing_policy"][Date("2020-01-01")]["household"]["symptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["household"]["asymptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["school"]["symptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["school"]["asymptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["workplace"]["symptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["workplace"]["asymptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["community"]["symptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["community"]["asymptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["social"]["symptomatic"],
    d["tracing_policy"][Date("2020-01-01")]["social"]["asymptomatic"],
    d["quarantine_policy"][Date("2020-01-01")]["awaiting_test_result"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["awaiting_test_result"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["tested_positive"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["tested_positive"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["household"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["household"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["school"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["school"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["workplace"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["workplace"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["community"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["community"]["compliance"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["social"]["days"],
    d["quarantine_policy"][Date("2020-01-01")]["case_contacts"]["social"]["compliance"]
    ]
end

"Take filtered file and write parameters and means to EOF"
function write_GSA(infile::String, ofile::String, configfile::String)
    indata = DataFrame(CSV.File(infile))
    data = groupby(indata,:run)

    odata = combine(data) do d
        # d.positives[dt]-d.positives[dt-1]
        (mnp,tmax) = findmax(d.positives[2:end]-d.positives[1:end-1])
        DataFrame(
            t_max = tmax,max_newpos=mnp,
            pos = d.positives[end],
            R_inf = d.R[end]
        )
    end
    a = describe(odata)
    

    params = get_params(configfile)
    writer = DataFrame(transpose([a.mean;params]))
    show(writer)
    CSV.write(ofile, writer, writeheader = false, append = true)
end

function GSAwrite(configfile)
    d = YAML.load_file(configfile)
    infile  = d["output_directory"] * "/metrics.csv"
    outfile = d["output_directory"] * "/metrics_filtered.csv"
    filter_result(infile, outfile, :address, "0")
    write_GSA(outfile,d["output_directory"] * "/GSA_summary.csv",configfile)
end

end