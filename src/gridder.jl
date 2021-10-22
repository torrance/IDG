using FFTW: fft, fftshift, ifftshift

function parallelgrid!(mastergrid, subgrids, c)
    ch = Channel{Tuple{Subgrid, Task}}(1000)
    Threads.@spawn begin
        for subgrid in subgrids.children
            task = Threads.@spawn gridder(subgrid, subgrids.gridspec, subgrids.subgridspec, c)
            put!(ch, (subgrid, task))
        end
        close(ch)
    end

    println("Beginning griding...")
    for (i, (subgrid, task)) in enumerate(ch)
        grid = fetch(task)
        print("\rProgress $(i)/$(length(subgrids.children))")
        departition!(mastergrid, grid, subgrid)
    end
    println("...Done")
end


function gridder(subgrid::Subgrid, gridspec, subgridspec, c)
    grid = zeros(ComplexF32, subgridspec.Nx, subgridspec.Ny)
    dft!(grid, subgrid, gridspec, subgridspec, c)
    return ifftshift(fft(ifftshift(grid))) ./ (subgridspec.Nx * subgridspec.Ny)
end   

function dft!(grid::Matrix{ComplexF32}, subgrid::Subgrid, gridspec::GridSpec, subgridspec::GridSpec, c)
    N = subgridspec.Nx
    lms = px2sky(N, subgridspec)

    u0px = subgrid.u0px + (subgridspec.Nx ÷ 2) # Central 'zeroth' pixel
    v0px = subgrid.v0px + (subgridspec.Ny ÷ 2)
    u0lambda, v0lambda = px2lambda(u0px, v0px, gridspec)

    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        n2 = 1 - l^2 - m^2
        if n2 >= 0
            ndash = sqrt(n2) - 1
            for uvdatum in subgrid.children
                phase = 2π * 1im * (
                    (uvdatum.u - u0lambda) * l +
                    (uvdatum.v - v0lambda) * m +
                    (uvdatum.w - subgrid.w0lambda) * ndash
                )
                grid[lpx, mpx] += 0.5 * (uvdatum.xx + uvdatum.yy) * exp(phase)
            end
            grid[lpx, mpx] *= c(l, m)
        end
    end
end