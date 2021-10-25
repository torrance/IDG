import Statistics: median

struct Subgrid
    u0px::Int  # px location of 1 pixel
    v0px::Int  # px location of 1 pixel
    w0lambda::Int32  # lambda
    children::Vector{UVDatum}
end

struct Subgrids
    gridspec::GridSpec
    subgridspec::GridSpec
    padding::Int
    children::Vector{Subgrid}
end

function Subgrid(u0px, v0px, w0lambda; children::Union{Nothing, Vector{UVDatum}}=nothing)
    if children === nothing
        children = UVDatum[]
    end
    return Subgrid(u0px, v0px, w0lambda, children)
end

function Subgrids(gridspec::GridSpec, subgridspec::GridSpec, padding::Int)
    Subgrids(gridspec, subgridspec, padding, Subgrid[])
end

function partition!(subgrids::Subgrids, uvdata)
    # Separate data in w layers and create UVData
    println("Creating w layers...")
    wlayers = Dict{Int, Array{UVDatum, 1}}()
    @time for uvdatum in uvdata
        wlayer = get!(wlayers, round(Int, uvdatum.w)) do
            UVDatum[]
        end
        push!(wlayer, uvdatum)
    end

    # Now distribute partitioning work to threads by w layer
    println("Partitioning...")
    @time begin
        tasks = Task[]
        for wlayer in values(wlayers)
            task = Threads.@spawn try
                partitionconsumer(wlayer, subgrids.gridspec, subgrids.subgridspec, subgrids.padding)
            catch e
               showerror(stdout, e, catch_backtrace())
            end
            push!(tasks, task)
        end
        for task in tasks
            append!(subgrids.children, fetch(task))
        end
    end

    occupancy = [length(subgrid.children) for subgrid in subgrids.children]
    println("Subgrids: $(length(occupancy)) Occupancy (min/median/max): $(minimum(occupancy))/$(median(occupancy))/$(maximum(occupancy))")
end

function partitionconsumer(wlayer, gridspec, subgridspec, padding)
    subgrids = Subgrid[]
    for uvdatum in wlayer
        _partitionconsumer!(subgrids, uvdatum, gridspec, subgridspec, padding)
    end
    return subgrids
end

function _partitionconsumer!(subgrids::Vector{Subgrid}, uvdatum, gridspec, subgridspec, padding)
    upx, vpx = lambda2px(uvdatum.u, uvdatum.v, gridspec)

    # Now check through existing subgrids to see if our uvdatum already overlaps
    for subgrid in subgrids
        if (
            subgrid.u0px - 0.5 + padding < upx <= subgrid.u0px - 0.5 + subgridspec.Nx - padding &&
            subgrid.v0px - 0.5 + padding < vpx <= subgrid.v0px - 0.5 + subgridspec.Nx - padding
        )
            push!(subgrid.children, uvdatum)
            return
        end
    end

    # If we made it to here, there's no valid subgrid, so create a new one with (upx, vpx)
    # at bottom left
    u0px = round(Int, upx - subgridspec.Nx ÷ 2)
    v0px = round(Int, vpx - subgridspec.Ny ÷ 2)

    push!(subgrids, Subgrid(u0px, v0px, round(Int32, uvdatum.w), UVDatum[uvdatum]))
end

function departition!(mastergrid, grid, subgrid::Subgrid)
    u0px = subgrid.u0px
    v0px = subgrid.v0px

    for (j, vpx) in enumerate(v0px:v0px + size(grid)[2] - 1)
        if 1 <= vpx <= size(mastergrid)[2]
            for (i, upx) in enumerate(u0px:u0px + size(grid)[1] - 1)
                if 1 <= upx <= size(mastergrid)[1]
                    mastergrid[upx, vpx] += grid[i, j]
                end
            end
        end
    end
end