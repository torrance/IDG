using StaticArrays

struct UVDatum
    u::Float32
    v::Float32
    w::Float32
    xx::ComplexF32
    xy::ComplexF32
    yx::ComplexF32
    yy::ComplexF32
end

function mkuvdata(mset::MeasurementSet; imageweighter=nothing, makepsf::Bool=false)
    if imageweighter === nothing
        imageweighter = (_, _) -> return SVector{4, Float32}(1, 1, 1, 1)
    end

    ch = Channel{UVDatum}(10000)
    @async try
        for row in mset
            _mkuvdata(ch, row, mset.lambdas, imageweighter, makepsf)
        end
        close(ch)
    catch e
        showerror(stdout, e, catch_backtrace())
    end
    return ch
end

@views function _mkuvdata(ch, row, lambdas, imageweighter, makepsf)
    data = MVector{4, ComplexF32}(0, 0, 0, 0)
    for (chan, lambda) in enumerate(lambdas)
        u = row.uvw[1] / lambda
        v = -row.uvw[2] / lambda
        w = row.uvw[3] / lambda

        if makepsf
            data .= 1
        else
            data .= row.data[:, chan]
        end

        # Apply weights and flags to image data
        imageweights = imageweighter(u, v)
        empty = true
        for pol in 1:4
            weight = (
                !row.flag[pol, chan] *
                !row.flagrow[] *
                row.weight[pol] *
                row.weightspectrum[pol, chan] *
                imageweights[pol]
            )
            if !isfinite(weight) || !isfinite(row.data[pol, chan])
                data[pol] = 0
            else
                empty = false
                data[pol] *= weight
            end
        end

        if empty
            continue
        end

        put!(ch, UVDatum(
            u, v, w, data[1], data[2], data[3], data[4]
        ))
    end
end