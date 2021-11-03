using FFTW
using IDGjl
using PyCall
using Test
using Unitful
using UnitfulAngles
using Interpolations

# Now do a direct FT of this data into the image plane
function dft!(dft, uvdata, gridspec, normfactor)
    rowscomplete = 0
    Threads.@threads for lmpx in CartesianIndices(dft)
        lpx, mpx = Tuple(lmpx)
        if lpx == size(dft)[2]
            rowscomplete += 1
            print("\r", rowscomplete / size(dft)[1] * 100)
        end

        l, m = IDGjl.px2sky(lpx, mpx, gridspec)
        ndash = sqrt(1 - l^2 - m^2) - 1

        val = zero(ComplexF64)
        for uvdatum in uvdata
            val += 0.5 * (uvdatum.xx + uvdatum.yy) * exp(
                2π * 1im * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * ndash)
            )
        end
        dft[lpx, mpx] = val / normfactor
    end
end

function convolutionalsample!(grid, gridspec, uvdata, kernel, width; uoffset=0, voffset=0)
    for uvdatum in uvdata
        upx, vpx = IDGjl.lambda2px(uvdatum.u - uoffset, uvdatum.v - voffset, gridspec)

        for ypx in axes(grid, 2)
            dypx = abs(ypx - vpx)
            if 0 <= dypx <= width
                for xpx in axes(grid, 1)
                    dxpx = abs(xpx - upx)
                    if 0 <= dxpx <= width
                        grid[xpx, ypx] += kernel(sqrt(dxpx^2 + dypx^2)) * 0.5 * (uvdatum.xx + uvdatum.yy)
                    end
                end
            end
        end
    end
end

function mkoversampledtaper(taper, gridspec, oversample)
    oversampledsubgrid = IDGjl.GridSpec(
        gridspec.Nx * oversample,
        gridspec.Ny * oversample,
        gridspec.scalelm,
        gridspec.scaleuv
    )
    tapergrid = zeros(oversampledsubgrid.Ny, oversampledsubgrid.Ny)
    @time Threads.@threads for lm in CartesianIndices(tapergrid)
        lpx, mpx = Tuple(lm)
        l, m = IDGjl.px2sky(lpx, mpx, oversampledsubgrid)
        tapergrid[lm] = taper(l, m)
    end
    @time Tapergrid = rfft(ifftshift(tapergrid)) / (gridspec.Nx * gridspec.Ny)
    midy = size(Tapergrid)[2] ÷ 2 + 1
    Tapergrid = real.(fftshift(Tapergrid, 2))[:, midy:end]

    println("Calculating rs...")
    rs = similar(Tapergrid, Float64)
    @time ((rs) -> for xy in CartesianIndices(rs)
        xpx, ypx = Tuple(xy) .- 1
        rs[xy] = sqrt(xpx^2 + ypx^2) / oversample
    end)(rs)

    # Remove duplicate entries
    println("Deduplicate...")
    @time idx = unique(i -> rs[i], eachindex(rs))
    rs, Taper = rs[idx], Tapergrid[idx]

    # Sort
    println("Sorting...")
    @time idx = sortperm(rs)
    rs, Taper = rs[idx], Taper[idx]

    return LinearInterpolation(rs, Taper)
end

# @testset "IDGjl.jl" begin
#     mset = IDGjl.MeasurementSet("/home/torrance/testdata/1215555160/1215555160.ms", chanstart=92, chanend=92)

#     masterpadding = round(Int, 0.7 * 4000 / 2)
#     Nsize = 4000 + 2 * masterpadding
#     gridspec = IDGjl.GridSpec(Nsize, Nsize, 20u"arcsecond")

#     # taper, padding = IDGjl.mkgaussiantaper(gridspec, 1e-2)
#     taper, padding = IDGjl.mkkbtaper(gridspec, 1e-4)
#     println("Kernel width $(padding)")

#     # taper = (l, m) -> 1
#     # kernelwidth = 0

#     println("Imageweight...")
#     @time imageweighter = IDGjl.Briggs(IDGjl.mkuvdata(mset, makepsf=true), gridspec, 0.)
#     # @time imageweighter = IDGjl.Natural(IDGjl.mkuvdata(mset, makepsf=true), gridspec)

#     println("Collecting uvdata into memory...")
#     uvdata = collect(IDGjl.mkuvdata(mset, imageweighter=imageweighter))
#     println(typeof(uvdata))

#     # println("DFT...")
#     # dft = zeros(ComplexF64, Nsize, Nsize)
#     # dft!(dft, uvdata, gridspec, imageweighter.normfactor[1])

#     subgrids = IDGjl.Subgrids(
#         gridspec,
#         IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv),
#         padding,
#     )

#     println("Mmwpartition....")
#     @time IDGjl.partition!(subgrids, IDGjl.mkuvdata(mset, imageweighter=imageweighter, makepsf=false))

#     subgrid = subgrids.children[3]
#     println("Subgrid has $(length(subgrid.children)) children")

#     mastergrid = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
#     IDGjl.parallelgrid!(mastergrid, subgrids, taper)

#     idg = fftshift(ifft(ifftshift(mastergrid))) * gridspec.Nx * gridspec.Ny / imageweighter.normfactor[1]
#     for lm in CartesianIndices(idg)
#         lpx, mpx = Tuple(lm)
#         l, m = IDGjl.px2sky(lpx, mpx, gridspec)
#         idg[lm] /= taper(l, m)
#     end

#     return idg[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
#     return dft, idg
#     return mastergrid, img, imageweighter

#     # TODO:
#     # Understand what the output should be with w-correction term multiplied through
#     # Double check w layer tolerance in wsclean/IDG
# end

@testset "Partitioning" begin
    # Create set of uvdatum with random u, v, coordinates and fixed w
    uvdata = IDGjl.UVDatum[]
    for (u, v) in zip(rand(Float64, 100000), rand(Float64, 100000))
        u = u * 100 - 50
        v = v * 100 - 50
        push!(uvdata, IDGjl.UVDatum(u, v, 0, 0, 0, 0, 0))
    end

    # Append to start special seed UVDatum
    uvdata[1] = IDGjl.UVDatum(20, 20, 0, 0, 0, 0, 0)

    gridspec = IDGjl.GridSpec(100, 100, scaleuv=1)
    subgridspec = IDGjl.GridSpec(64, 64, scaleuv=1)
    padding = 8

    subgrids = IDGjl.Subgrids(gridspec, subgridspec, padding)
    IDGjl.partition!(subgrids, uvdata, 1)

    subgrid = subgrids.children[1]

    @test subgrid.u0px == 39
    @test subgrid.v0px == 39

    us = [uvdatum.u for uvdatum in subgrid.children]
    vs = [uvdatum.v for uvdatum in subgrid.children]

    @test 23.9 < maximum(us .- 19.5) <= 24
    @test -23.9 > minimum(us .- 19.5) > -24
    @test 23.9 < maximum(vs .- 19.5) <= 24
    @test -23.9 > minimum(vs .- 19.5) > -24
end

@testset "Departition" begin
    subgrid = IDGjl.Subgrid(346, 346, 0)

    expected = zeros(ComplexF64, 1000, 1000)
    grid = rand(ComplexF64, 64, 64)
    expected[346:346 + 63, 346:346 + 63] .= grid

    master = zeros(ComplexF64, 1000, 1000)
    IDGjl.departition!(master, grid, subgrid)

    @test all(master .== expected)

    # Negative grid
    subgrid = IDGjl.Subgrid(-20, 346, 0)

    expected = zeros(ComplexF64, 1000, 1000)
    grid = rand(ComplexF64, 64, 64)
    expected[1:1 + 63 - 21, 346:346 + 63] .= grid[22:end, :]

    master = zeros(ComplexF64, 1000, 1000)
    IDGjl.departition!(master, grid, subgrid)

    @test all(master .== expected)
end

@testset "PerfectGrid" begin
    #####
    # Test subrid inversion
    gridspec = IDGjl.GridSpec(64, 64, scaleuv=1)
    subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    # taper, padding = IDGjl.mkkbtaper(gridspec, 1e-5)
    taper, padding, sigmalm = IDGjl.mkgaussiantaper(gridspec, 1e-12)
    Taper = (rpx) -> 2π * (sigmalm * gridspec.scaleuv)^2 * exp(-2 * π^2 * (sigmalm * gridspec.scaleuv)^2 * rpx^2)

    expected = zeros(ComplexF32, 64, 64)
    expected[1 + padding:end - padding, 1 + padding:end - padding] = rand(Float32, gridspec.Nx - 2 * padding, gridspec.Ny - 2 * padding) .+ 1im .* rand(Float32, gridspec.Nx - 2 * padding, gridspec.Ny - 2 * padding)

    # Set some as zero, i.e. missing UV coverage
    expected[rand([true, false, false], 64, 64)] .= 0

    uvdata = IDGjl.UVDatum[]
    for uv in CartesianIndices(expected)
        upx, vpx = Tuple(uv)
        u, v = IDGjl.px2lambda(upx, vpx, gridspec)
        val = expected[uv]
        if val != 0
            push!(uvdata, IDGjl.UVDatum(u, v, 0, val, 0, 0, val))
        end
    end

    # Taper =  mkoversampledtaper(taper, gridspec, 200)
    expectedconv = zeros(ComplexF64, 64, 64)
    convolutionalsample!(expectedconv, subgridspec, uvdata, Taper, padding)

    subgrid = IDGjl.Subgrid(1, 1, 0, uvdata)
    grid = IDGjl.gridder(subgrid, gridspec, subgridspec, taper)

    @test all(isapprox.(expectedconv, grid, atol=1E-6))

    #####
    # Large composite grid
    gridspec = IDGjl.GridSpec(400, 400, scaleuv=1)
    subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    # taper, padding = IDGjl.mkkbtaper(gridspec, 1e-6)
    taper, padding, sigmalm = IDGjl.mkgaussiantaper(gridspec, 1e-12)
    Taper = (rpx) -> 2π * (sigmalm * gridspec.scaleuv)^2 * exp(-2 * π^2 * (sigmalm * gridspec.scaleuv)^2 * rpx^2)

    expected = rand(Float32, gridspec.Nx, gridspec.Ny) .+ 1im .* rand(Float32, gridspec.Nx, gridspec.Ny)
    # Set some as zero, i.e. missing UV coverage
    expected[rand([true, false, false, false, false, false, false, false], 400, 400)] .= 0

    # Create uvdata
    uvdata = IDGjl.UVDatum[]
    for uv in CartesianIndices(expected)
        upx, vpx = Tuple(uv)
        u, v = IDGjl.px2lambda(upx, vpx, gridspec)
        val = expected[upx, vpx]
        if val != 0
            push!(uvdata, IDGjl.UVDatum(u, v, 0, val, 0, 0, val))
        end
    end

    # Taper = mkoversampledtaper(taper, subgridspec, 200)
    expectedconv = zeros(ComplexF32, gridspec.Nx, gridspec.Ny)
    convolutionalsample!(expectedconv, gridspec, uvdata, Taper, padding)

    # IDG
    subgrids = IDGjl.Subgrids(gridspec, subgridspec, padding)
    IDGjl.partition!(subgrids, uvdata, 1)

    mastergrid = zeros(ComplexF32, gridspec.Nx, gridspec.Ny)
    for subgrid in subgrids.children
        grid = IDGjl.gridder(subgrid, gridspec, subgridspec, taper)
        IDGjl.departition!(mastergrid, grid, subgrid)
    end

    @test all(isapprox.(expectedconv, mastergrid, atol=3E-6))
end

@testset "Coordinate conversions" begin
    for (ulambda, vlambda) in eachcol(100 .* rand(2, 100))
        gridspec = IDGjl.GridSpec(3000, 3000, scaleuv=rand() + 0.5)
        upx, vpx = IDGjl.lambda2px(ulambda, vlambda, gridspec)
        ulambdaagain, vlambdaagain = IDGjl.px2lambda(upx, vpx, gridspec)
        @test ulambda ≈ ulambdaagain
        @test vlambda ≈ vlambdaagain
    end
end

@testset "Imperfect grid" begin
    # Create uvw data with a few sources, and uvw points randomly in a cube
    uvws = rand(Float32, 3, 2000) .* [1000 1000 0]' .- [500 500 0;]'
    uvws .= round.(Int, uvws)

    # Source locations in radians, wrt to phase center, with 10 degree FOV
    sources = deg2rad.(
        rand(Float32, 2, 30) * 18 .- 9
    )
    sources[:, 1] .= 0

    uvdata = IDGjl.UVDatum[]
    for (u, v, w) in eachcol(uvws)
        val = zero(ComplexF32)
        for (ra, dec) in eachcol(sources)
            l, m = sin(ra), sin(dec)
            val += exp(-2π * 1im * (u * l + v * m + w * IDGjl.ndash(l, m)))
        end
        push!(uvdata, IDGjl.UVDatum(u, v, w, val, 0, 0, val))
    end

    # IDG
    gridspec = IDGjl.GridSpec(3000, 3000, scaleuv=0.6)
    subgridspec = IDGjl.GridSpec(128, 128, scaleuv=gridspec.scaleuv)
    taper, padding, sigmalm = IDGjl.mkgaussiantaper(gridspec, 1e-12)
    Taper = (rpx) -> 2π * (sigmalm * gridspec.scaleuv)^2 * exp(-2 * π^2 * (sigmalm * gridspec.scaleuv)^2 * rpx^2)
    println("Taper with $(padding) padding")

    subgrids = IDGjl.Subgrids(gridspec, subgridspec, padding)
    IDGjl.partition!(subgrids, uvdata, 1)
    println("Original uv data points: $(length(uvdata)) After partitioning: $(sum(length(subgrid.children) for subgrid in subgrids.children))")

    # Compare single subgrid
    _, idx = findmax([length(subgrid.children) for subgrid in subgrids.children])
    subgrid = subgrids.children[idx]
    subgridded = IDGjl.gridder(subgrid, gridspec, subgridspec, taper)

    uoffset, voffset = IDGjl.px2lambda(
        subgrid.u0px + (subgridspec.Nx ÷ 2),
        subgrid.v0px + (subgridspec.Ny ÷ 2),
        gridspec
    )
    expected = zeros(ComplexF64, subgridspec.Nx, subgridspec.Ny)
    convolutionalsample!(expected, subgridspec, subgrid.children, Taper, 2 * padding, uoffset=uoffset, voffset=voffset)

    @test all(isapprox.(expected, subgridded, atol=5e-7))

    mastergrid = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
    IDGjl.parallelgrid!(mastergrid, subgrids, taper)

    # Manual convolutional sample
    expected = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
    convolutionalsample!(expected, gridspec, uvdata, Taper, padding)

    @test all(isapprox.(expected, mastergrid, atol=5e-7))
end

# @testset "Imperfect grid with w terms" begin
#     Nsize = 2000

#     # Create uvw data with a few sources, and uvw points randomly in a cube
#     # uvws = cat(
#     #     rand(Float32, 3, 1500) .* [1000 500 0;]' .+ [-500 0 20.5;]',
#     #     rand(Float32, 3, 1500) .* [1000 -500 0;]' .+ [-500 0 -14.5;]',
#     # dims=(2))
#     uvws = rand(Float32, 3, 5000) .* [1000 1000 0;]' .- [500 500 (-0.5);]'
#     # vuvws = zeros(3, 0)
#     # for n in 1.1:.1:2
#     #     uvws = cat(uvws, _uvws ./ n, dims=2)
#     # end
#     # uvws[3, :] .= 100
#     println(size(uvws))

#     # Source locations in radians, wrt to phase center, with 10 degree FOV
#     sources = deg2rad.(
#         rand(Float32, 2, 50) * 18 .- 9
#     )
#     sources[:, 1] .= 0
#     # sources = Float32[0 0;]'

#     uvdata = IDGjl.UVDatum[]
#     for (u, v, w) in eachcol(uvws)
#         val = zero(ComplexF32)
#         for (ra, dec) in eachcol(sources)
#             l, m = sin(ra), sin(dec)
#             ndash = sqrt(1 - l^2 - m^2) - 1
#             val += exp(-2π * 1im * (u * l + v * m + w * ndash))
#         end
#         push!(uvdata, IDGjl.UVDatum(u, v, w, val, 0, 0, val))
#     end

#     print("Running DFT...")
#     gridspec = IDGjl.GridSpec(Nsize, Nsize, 40u"arcsecond")
#     @assert maximum(uvws[1, :]) < IDGjl.px2lambda(gridspec.Nx, gridspec.Ny ÷ 2 + 1, gridspec)[1]
#     @assert maximum(uvws[2, :]) < IDGjl.px2lambda(gridspec.Nx ÷ 2 + 1, gridspec.Ny, gridspec)[2]
#     @assert minimum(uvws[1, :]) > IDGjl.px2lambda(1, gridspec.Ny ÷ 2 + 1, gridspec)[1]
#     @assert minimum(uvws[2, :]) > IDGjl.px2lambda(gridspec.Nx ÷ 2 + 1, 1, gridspec)[2]
#     dft = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
#     @time dft!(dft, uvdata, gridspec, length(uvdata))
#     println(" done.")
#     println("Max value in DFT: $(maximum(real.(dft)))")

#     # Now it's IDG time
#     masterpadding = round(Int, (Nsize / 0.6) / 2)
#     gridspec = IDGjl.GridSpec(Nsize + 2 * masterpadding, Nsize + 2 * masterpadding, 40u"arcsecond")
#     taper, padding = IDGjl.mkkbtaper(gridspec, 1e-4)
#     println("Using kernel padding $(padding)")
#     # taper, padding = (l, m) -> 1, 1
#     # taper, padding = IDGjl.mkgaussiantaper(gridspec, 1e-5)
#     subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
#     subgrids = IDGjl.Subgrids(gridspec, subgridspec, padding)

#     println("subgridspec edges:", subgridspec.scaleuv * 63)


#     IDGjl.partition!(subgrids, uvdata, 50)
#     println("Original uv data points: $(length(uvdata)) After partitioning: $(sum(length(subgrid.children) for subgrid in subgrids.children))")

#     for subgrid in subgrids.children
#         println(subgrid.w0lambda)
#     end

#     mastergrid = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
#     println("Gridding...")
#     @time IDGjl.parallelgrid!(mastergrid, subgrids, taper)
#     return mastergrid

#     println("FFT...")
#     @time idg = fftshift(ifft(ifftshift(mastergrid))) * gridspec.Nx * gridspec.Ny / length(uvdata)

#     function removetaper!(img, gridspec, taper)
#         Threads.@threads for lm in CartesianIndices(img)
#             lpx, mpx = Tuple(lm)
#             l, m = IDGjl.px2sky(lpx, mpx, gridspec)
#             n2 = 1 - l^2 - m^2
#             if n2 < 0
#                 img[lm] = NaN
#             else
#                 ndash = sqrt(n2) - 1
#                 img[lm] *= 1 / taper(l, m) # exp(2π * 1im * 100 * ndash) / taper(l, m) #
#             end
#         end
#     end
#     println("Removing taper...")
#     @time removetaper!(idg, gridspec, taper)

#     println("Max value in IDG: $(maximum(real.(idg)))")

#     return dft, idg[1+masterpadding:end-masterpadding, 1+masterpadding:end-masterpadding]
# end
