using FFTW: fft, ifft, fftshift, ifftshift
using IDGjl
using PyCall
using Test
using Unitful
using UnitfulAngles

# @testset "IDGjl.jl" begin
#     mset = IDGjl.MeasurementSet("/home/torrance/testdata/1215555160/1215555160.ms", chanstart=1, chanend=196)
    
#     Nsize = 4000
#     gridspec = IDGjl.GridSpec(Nsize, Nsize, 20u"arcsecond")

#     taper, kernelwidth = IDGjl.mkgaussiantaper(gridspec, 1e-2)
#     println("Kernel width $(kernelwidth)")

#     # taper = (l, m) -> 1
#     # kernelwidth = 0

#     println("Imageweight...")
#     # @time imageweighter = IDGjl.Briggs(IDGjl.mkuvdata(mset, makepsf=true), gridspec, 0.)
#     @time imageweighter = IDGjl.Natural(IDGjl.mkuvdata(mset, makepsf=true), gridspec)

#     subgrids = IDGjl.Subgrids(
#         gridspec,
#         IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv),
#         kernelwidth,
#         IDGjl.Subgrid[],
#     )

#     println("Mmwpartition....")
#     @time IDGjl.partition!(subgrids, IDGjl.mkuvdata(mset, imageweighter=imageweighter, makepsf=false), imageweighter)

#     subgrid = subgrids.children[3]
#     println("Subgrid has $(length(subgrid.children)) children")

#     subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
#     mastergrid = zeros(ComplexF64, Nsize, Nsize)
#     IDGjl.parallelgrid!(mastergrid, subgrids, taper)

#     img = fftshift(ifft(ifftshift(mastergrid))) * gridspec.Nx * gridspec.Ny / imageweighter.normfactor[1]
#     for lm in CartesianIndices(img)
#         lpx, mpx = Tuple(lm)
#         l, m = IDGjl.px2sky(lpx, mpx, gridspec)
#         img[lm] /= taper(l, m)
#     end

#     return mastergrid, img, imageweighter

#     # TODO:
#     # Understand what the output should be with w-correction term multiplied through
#     # Double check w layer tolerance in wsclean/IDG
# end

# @testset "Gridder" begin
#     Ngrid = 64
#     gridspec = IDGjl.GridSpec(4000, 4000, 20u"arcsecond")
#     subgridspec = IDGjl.GridSpec(Ngrid, Ngrid, scaleuv=gridspec.scaleuv)

#     function makec(wlambda)
#         c = zeros(ComplexF32, Ngrid, Ngrid)
#         for mpx in 1:Ngrid, lpx in 1:Ngrid
#             lpxzeroed = lpx - (Ngrid ÷ 2 + 1)
#             mpxzeroed = mpx - (Ngrid ÷ 2 + 1)
#             l, m = IDGjl.px2sky(lpx, mpx, subgridspec)
#             n2 = 1 - l^2 - m^2
#             ndash = 0.
#             if n2 >= 0
#                 ndash = sqrt(n2) - 1
#             end
#             ndash = sqrt(n2) - 1
#             c[lpx, mpx] = exp(-(lpxzeroed^2 + mpxzeroed^2) / (2 * 9^2)) * exp(2π * 1im * wlambda * ndash)
#         end
#         return c
#     end

#     mastergrid = zeros(ComplexF32, Ngrid, Ngrid)
#     expected = zeros(Complex, Ngrid, Ngrid)
#     for w0lambda in [0, 3, 7, 9, 11, 13, 44, 29, 21, 91]
#         subgrid = IDGjl.Subgrid(2000, 2000, w0lambda)

#         visgrid = zeros(ComplexF32, Ngrid, Ngrid)
#         visgrid[9:end-8, 9:end-8] = rand([0, 0, 0, 0, 0, 0, 1], Ngrid - 16, Ngrid - 16)

#         for uv in CartesianIndices(visgrid)
#             upx, vpx = Tuple(uv)
#             if visgrid[upx, vpx] != 0
#                 ulambda, vlambda = IDGjl.px2lambda(subgrid.u0px + 0.5 + upx - subgridspec.Nx ÷ 2 - 1, subgrid.v0px + 0.5 + vpx - subgridspec.Nx ÷ 2 - 1, gridspec)
#                 val = 0.0 + 0.0im
#                 for (A, l, m) in [(1, 0, 0), (2, sin(deg2rad(2)), sin(deg2rad(2))), (3, sin(deg2rad(-2)), sin(deg2rad(-2)))]
#                     val += A * exp(-2π * 1im * (ulambda * l + vlambda * m + w0lambda * (sqrt(1 - l^2 -m^2) - 1)))
#                 end
#                 push!(subgrid.children, IDGjl.UVDatum(ulambda, vlambda, w0lambda, val, 0, 0, val))
#             end
#         end
        
#         mastergrid += IDGjl.gridder(subgrid, gridspec, subgridspec, makec(w0lambda))

#         subgrid = IDGjl.Subgrid(2000, 2000, 0, subgrid.children) # Inherit same children
#         grid = zeros(ComplexF32, Ngrid, Ngrid)
#         IDGjl.dft!(grid, subgrid, gridspec, subgridspec, makec(0))
#         expected += grid
#     end

#     # TODO Understand the indexing offset here
#     diff = expected[end:-1:2, end:-1:2] .- fftshift(fft(ifftshift(mastergrid)))[2:end, 2:end]
#     @test all(isapprox.(diff, 0, atol=1e-5))
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
    IDGjl.partition!(subgrids, uvdata)

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
    onef = (l, m) -> 1.

    expected = zeros(ComplexF32, 64, 64)
    expected[1 + 8:end - 8, 1 + 8:end - 8] = rand(Float32, gridspec.Nx - 16, gridspec.Ny - 16) .+ 1im .* rand(Float32, gridspec.Nx - 16, gridspec.Ny - 16)
    
    # Set some as zero, i.e. missing UV coverage
    expected[rand([true, false, false], 64, 64)] .= 0
    
    subgrid = IDGjl.Subgrid(1, 1, 0, IDGjl.UVDatum[])
    for uv in CartesianIndices(expected)
        upx, vpx = Tuple(uv)
        u, v = IDGjl.px2lambda(upx, vpx, gridspec)
        val = expected[uv]
        if val != 0
            push!(subgrid.children, IDGjl.UVDatum(u, v, 0, val, 0, 0, val))
        end
    end
    grid = IDGjl.gridder(subgrid, gridspec, subgridspec, onef)

    @test all(isapprox.(expected, grid, atol=1E-5))

    #####
    # Large composite grid
    gridspec = IDGjl.GridSpec(400, 400, scaleuv=1)
    subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)

    expected = rand(Float32, gridspec.Nx, gridspec.Ny) .+ 1im .* rand(Float32, gridspec.Nx, gridspec.Ny)

    # Create uvdata
    uvdata = IDGjl.UVDatum[]
    for uv in CartesianIndices(expected)
        upx, vpx = Tuple(uv)
        u, v = IDGjl.px2lambda(upx, vpx, gridspec)
        val = expected[upx, vpx]
        push!(uvdata, IDGjl.UVDatum(u, v, 0, val, 0, 0, val))
    end

    # Partition data into subgrids
    subgrids = IDGjl.Subgrids(gridspec, subgridspec, 8)
    IDGjl.partition!(subgrids, uvdata)

    # Check all UVDatum exist in exactly one partition
    mastergrid = zeros(Int, 400, 400)
    for subgrid in subgrids.children, uvdatum in subgrid.children
        upx, vpx = IDGjl.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        mastergrid[upx, vpx] += 1
    end
    @test all(mastergrid .== 1)

    mastergrid = zeros(ComplexF32, gridspec.Nx, gridspec.Ny)
    for subgrid in subgrids.children
        grid = IDGjl.gridder(subgrid, gridspec, subgridspec, onef)
        IDGjl.departition!(mastergrid, grid, subgrid)
    end

    @test all(isapprox.(expected, mastergrid, atol=1E-5))
end

# @testset "Imperfect grid with w terms" begin
#     # Create uvw data with a few sources, and uvw points randomly in a cube
#     uvws = rand(Float32, 3, 2000) .* [500 500 0;]' .- [250 250 250;]'

#     # Source locations in radians, wrt to phase center, with 10 degree FOV
#     sources = deg2rad.(
#         rand(Float32, 2, 30) * 18 .- 9
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

#     # Now do a direct FT of this data into the image plane
#     function dft!(dft, uvdata, gridspec)
#         Threads.@threads for lmpx in CartesianIndices(dft)
#             lpx, mpx = Tuple(lmpx)
#             l, m = IDGjl.px2sky(lpx, mpx, gridspec)
#             ndash = sqrt(1 - l^2 - m^2) - 1

#             val = zero(ComplexF64)
#             for uvdatum in uvdata
#                 val += 0.5 * (uvdatum.xx + uvdatum.yy) * exp(
#                     2π * 1im * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * ndash)
#                 )
#             end
#             dft[lpx, mpx] = val / size(uvws)[2]
#         end
#     end

#     print("Running DFT...")
#     gridspec = IDGjl.GridSpec(2000, 2000, 40u"arcsecond")
#     @assert maximum(uvws[1, :]) < IDGjl.px2lambda(gridspec.Nx, gridspec.Ny ÷ 2 + 1, gridspec)[1]
#     @assert maximum(uvws[2, :]) < IDGjl.px2lambda(gridspec.Nx ÷ 2 + 1, gridspec.Ny, gridspec)[2]
#     @assert minimum(uvws[1, :]) > IDGjl.px2lambda(1, gridspec.Ny ÷ 2 + 1, gridspec)[1]
#     @assert minimum(uvws[2, :]) > IDGjl.px2lambda(gridspec.Nx ÷ 2 + 1, 1, gridspec)[2]
#     dft = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
#     dft!(dft, uvdata, gridspec)
#     println(" done.")
#     println("Max value in DFT: $(maximum(real.(dft)))")

#     # Now it's IDG time
#     c = (l, m) -> 1
#     taper, kernelwidth = IDGjl.mkgaussiantaper(gridspec, 1e-2)
#     subgridspec = IDGjl.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
#     subgrids = IDGjl.Subgrids(gridspec, subgridspec, kernelwidth, IDGjl.Subgrid[])
    
#     wlayers = Dict{Int, Array{IDGjl.UVDatum, 1}}()
#     for uvdatum in uvdata
#         wlayer = get!(wlayers, round(Int, uvdatum.w)) do 
#             IDGjl.UVDatum[]
#         end
#         push!(wlayer, uvdatum)
#     end
#     for wlayer in values(wlayers)
#         append!(
#             subgrids.children,
#             IDGjl.partitionconsumer(wlayer, subgrids.gridspec, subgrids.subgridspec, subgrids.padding)
#         )
#     end
#     println("Original uv data points: $(length(uvdata)) After partitioning: $(sum(length(subgrid.children) for subgrid in subgrids.children))")

#     mastergrid = zeros(ComplexF64, gridspec.Nx, gridspec.Ny)
#     IDGjl.parallelgrid!(mastergrid, subgrids, taper)
#     idg = fftshift(ifft(ifftshift(mastergrid))) * gridspec.Nx * gridspec.Ny / length(uvdata)

#     for lm in CartesianIndices(idg)
#         lpx, mpx = Tuple(lm)
#         l, m = IDGjl.px2sky(lpx, mpx, gridspec)
#         idg[lm] /= taper(l, m)
#     end

#     println("Max value in IDG: $(maximum(real.(idg)))")

#     return dft, idg
# end
