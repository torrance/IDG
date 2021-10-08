mutable struct CasaCol{T}
    col::PyCall.PyObject
    nrows::Int
    offset::Int
    ncache::Int
    cache::Union{Nothing, T}
    blc::Union{Nothing, Tuple{Int, Int}}
    trc::Union{Nothing, Tuple{Int, Int}}
end

function CasaCol(col::PyCall.PyObject, T::Type; blc=nothing, trc=nothing)
    return CasaCol{T}(col, col.nrows(), -1, 10000, nothing, blc, trc)
end

struct CasaColBoundsError <: Exception
    i::Int
    nrows::Int
end

struct CasaRowBoundsError <: Exception
    i::Int
    nrows::Int
end

struct MeasurementSet
    tbl::PyCall.PyObject
    chanstart::Int
    chanend::Int
    nrows::Int
    freqs::Array{Float32}
    lambdas::Array{Float32}
    # antennas::Dict{Int, Tuple{String, @NamedTuple{x::Float64, y::Float64, z::Float64}}}
    phasecenter::NamedTuple{(:ra, :dec), Tuple{Float64, Float64}}
    uvw::CasaCol
    ant1::CasaCol
    ant2::CasaCol
    flagrow::CasaCol
    flag::CasaCol
    weight::CasaCol
    weightspectrum::CasaCol
    data::CasaCol 
end

# TODO: Allow spw selection
function MeasurementSet(path::String; datacol=nothing, chanstart=0, chanend=0, autocorrelations=false, flagrow=false)
    PyCasaTable = PyCall.pyimport("casacore.tables")
    tbl = PyCasaTable.table(path)

    # Remove autocorrelations and flagrows
    conditions = String[]
    if !autocorrelations
        push!(conditions, "ANTENNA1 <> ANTENNA2")
    end
    if !flagrow
        push!(conditions, "not FLAG_ROW")
    end
    if length(conditions) != 0
        tbl = PyCasaTable.taql("select * from \$1 where " * join(conditions, " and "), tables=[tbl])
    end

    freqs = tbl.SPECTRAL_WINDOW.getcellslice("CHAN_FREQ", 0, chanstart - 1, chanend - 1)
    lambdas = c ./ freqs
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[1, :]
    if datacol === nothing
        datacol = "CORRECTED_DATA" in tbl.colnames() ? "CORRECTED_DATA" :  "DATA"
    end

    return MeasurementSet(
        tbl,
        chanstart,
        chanend,
        tbl.nrows(),
        freqs,
        lambdas,
        (ra=ra0, dec=dec0),
        CasaCol(tbl.col("UVW"), Array{Float32, 2}),
        CasaCol(tbl.col("ANTENNA1"), Array{Int, 1}),
        CasaCol(tbl.col("ANTENNA2"), Array{Int, 1}),
        CasaCol(tbl.col("FLAG_ROW"), Array{Bool, 1}),
        CasaCol(tbl.col("FLAG"), Array{Bool, 3}, blc=(chanstart - 1, -1), trc=(chanend - 1, -1)),
        CasaCol(tbl.col("WEIGHT"), Array{Float32, 2}),
        CasaCol(tbl.col("WEIGHT_SPECTRUM"), Array{Float32, 3}, blc=(chanstart - 1, -1), trc=(chanend - 1, -1)),
        CasaCol(tbl.col(datacol), Array{ComplexF32, 3}, blc=(chanstart - 1, -1), trc=(chanend - 1, -1)),
    )
end

@inline function Base.getindex(col::CasaCol, i::Int)
    @boundscheck checkbounds(col, i)

    # Check whether we have this chunk of the column precached
    if col.cache === nothing || !(col.offset <= i < col.offset + col.ncache)
        # Note: startrow in the pycall is zero indexed
        if col.blc === nothing && col.trc === nothing
            pydata = PyCall.pycall(col.col.getcol, PyCall.PyArray, startrow=i - 1, nrow=col.ncache)
        else
            pydata = PyCall.pycall(col.col.getcolslice, PyCall.PyArray, col.blc, col.trc, startrow=i - 1, nrow=col.ncache)
        end
        col.cache = permutedims(pydata, ndims(pydata):-1:1)
        col.offset = i
    end

    return selectdim(col.cache, ndims(col.cache), i - col.offset + 1)
end

function Base.checkbounds(col::CasaCol, i::Int)
    if !(1 <= i <= length(col))
        throw(CasaColBoundsError(i, length(col)))
    end
end

function Base.showerror(io::IO, e::CasaColBoundsError)
    print(io, "Attempted to access $(e.nrows)-element column at index [$(e.i)]")
end

@inline function Base.iterate(col::CasaCol)
    if length(col) == 0
        return nothing
    end
    return (@inbounds col[1], 2)
end

@inline function Base.iterate(col::CasaCol, i)
    if i > length(col)
        return nothing
    end
    return (@inbounds col[i], i + 1)
end

@inline function Base.length(col::CasaCol)
    return col.nrows
end

@inline function Base.getindex(mset::MeasurementSet, i::Int)
    @boundscheck checkbounds(mset, i)
    @inbounds return (
        uvw=mset.uvw[i],
        ant1=mset.ant1[i],
        ant2=mset.ant2[i],
        flagrow=mset.flagrow[i],
        flag=mset.flag[i],
        weight=mset.weight[i],
        weightspectrum=mset.weightspectrum[i],
        data=mset.data[i],
    )
end

function Base.checkbounds(mset::MeasurementSet, i::Int)
    if !(1 <= i <= length(mset))
        throw(CasaRowBoundsError(i, length(mset)))
    end
end

function Base.showerror(io::IO, e::CasaRowBoundsError)
    print(io, "Attempted to access $(e.nrows)-element row at index [$(e.i)]")
end

@inline function Base.iterate(mset::MeasurementSet)
    if length(mset) == 0
        return nothing
    end
    return (@inbounds mset[1], 2)
end

@inline function Base.iterate(mset::MeasurementSet, i)
    if i > length(mset)
        return nothing
    end
    return (@inbounds mset[i], i + 1)
end

@inline function Base.length(mset::MeasurementSet)
    return mset.nrows
end