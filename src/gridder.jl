using FFTW: fft, fftshift, ifftshift, fftfreq

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
    grid = zeros(ComplexF64, subgridspec.Nx, subgridspec.Ny)
    dft!(grid, subgrid, gridspec, subgridspec, c)
    return fftshift(fft(grid)) ./ (subgridspec.Nx * subgridspec.Ny)
end

function dft!(grid::Matrix{ComplexF64}, subgrid::Subgrid, gridspec::GridSpec, subgridspec::GridSpec, c)
    lms = fftfreq(subgridspec.Nx, 1 / subgridspec.scaleuv)

    u0px = subgrid.u0px + (subgridspec.Nx ÷ 2) # Central 'zeroth' pixel
    v0px = subgrid.v0px + (subgridspec.Ny ÷ 2)
    u0lambda, v0lambda = px2lambda(u0px, v0px, gridspec)

    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        for uvdatum in subgrid.children
            phase = 2π * 1im * (
                (uvdatum.u - u0lambda) * l +
                (uvdatum.v - v0lambda) * m +
                (uvdatum.w - subgrid.w0lambda) * ndash(l, m)
            )
            grid[lpx, mpx] += 0.5 * (uvdatum.xx + uvdatum.yy) * exp(phase)
        end
        grid[lpx, mpx] *= c(l, m)
    end
end

"""
    Calculate n' = 1 - n
                 = 1 - √(1 - l^2 - m^2)
                 = (l^2 + m^2) / (1 + √(1 - l^2 - m^2))

    The final line is (apparently) more accurate for small values of n.

    Note for values of n > 1, which are unphysical, ndash is set to 1.
"""
@inline function ndash(l, m)
    r2 = l^2 + m^2
    if r2 > 1
        return 1
    else
        return r2 / (1 + sqrt(1 - r2))
    end
end