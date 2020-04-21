module utils

using CSV
using DataFrames

function import_data(tablename2datafile::Dict{String, String})
    result = Dict{String, DataFrame}()
    for (tablename, datafile) in tablename2datafile
        result[tablename] = DataFrame(CSV.File(datafile))
    end
    result
end

function dict_to_namedtuple(d::Dict{Symbol, V}) where V
    (; zip(Tuple(collect(keys(d))), Tuple(collect(values(d))))...)
end

end