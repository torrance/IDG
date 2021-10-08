import Statistics: median

struct Subgrid
    u0px::Float32  # px
    v0px::Float32  # px
    w0lambda::Int32  # lambda
    N::Int32
    children::Array{UVDatum}
end

function partition(mset::MeasurementSet, gridspec, kernelwidth, imageweighter)
    # Separate data in w layers and create UVData
    wlayers = Dict{Int, Array{UVDatum, 1}}()
    println("Creating w layers...")
    @time for row in mset
        wlayerer!(wlayers, row, mset.lambdas, imageweighter)
    end

    # Now distribute partitioning work to threads by w layer
    println("Partitioning...")
    @time begin
        tasks = Task[]
        for wlayer in values(wlayers)
            task = Threads.@spawn try
                partitionconsumer(wlayer, gridspec, kernelwidth)
            catch e
                showerror(stdout, e, catch_backtrace())
            end
            push!(tasks, task)
        end
        subgrids = vcat([fetch(task) for task in tasks]...)
    end

    occupancy = [length(subgrid.children) for subgrid in subgrids]
    println("Subgrids: $(length(occupancy)) Occupancy (min/median/max): $(minimum(occupancy))/$(median(occupancy))/$(maximum(occupancy))")
    return subgrids
end

function wlayerer!(wlayers, row, lambdas, imageweighter)
    for (chan, lambda) in enumerate(lambdas)
        u = row.uvw[1] / lambda
        v = row.uvw[2] / lambda
        w = row.uvw[3] / lambda

        # Apply wieghts and flags to image data
        imageweights = imageweighter(u, v)
        for pol in 1:4
            row.data[pol, chan] *= (
                row.flag[pol, chan] * 
                !row.flagrow[] *
                row.weight[pol] *
                row.weightspectrum[pol, chan] *
                imageweights[pol]
            )
        end

        wlayer = get!(wlayers, round(Int, w)) do 
            UVDatum[]
        end
        push!(wlayer, UVDatum(
            u, v, w, row.data[1, chan], row.data[2, chan], row.data[3, chan], row.data[4, chan]
        ))
    end
end

function partitionconsumer(wlayer, gridspec, kernelwidth)
    subgrids = Subgrid[]
    for uvdatum in wlayer
        _partitionconsumer(subgrids, uvdatum, gridspec, kernelwidth)
    end
    return subgrids
end

function _partitionconsumer(subgrids, uvdatum, gridspec, kernelwidth)
    upx, vpx = lambda2px(uvdatum.u, uvdatum.v, gridspec)

    # Now check through existing subgrids to see if our uvdatum already overlaps
    for subgrid in subgrids
        udelt = subgrid.u0px - upx
        vdelt = subgrid.v0px - vpx
        if -kernelwidth < udelt <= kernelwidth && -kernelwidth < vdelt <= kernelwidth
            push!(subgrid.children, uvdatum)
            return
        end
    end

    # If we made it to here, there's no valid subgrid, so create a new one.
    u0px = round(upx - 0.5f0) + 0.5f0  # Round to nearest 0.5
    v0px = round(vpx - 0.5f0) + 0.5f0

    push!(subgrids, Subgrid(u0px, v0px, round(Int32, uvdatum.w), kernelwidth, UVDatum[uvdatum]))
end