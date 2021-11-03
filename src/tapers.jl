using SpecialFunctions: besseli
using FFTW: fft, ifft, fftshift, ifftshift

"""
    For a given gridspec, find the maximally sized Gaussian that reduces to threshold mu
    by the edge.

    Returns the taper function c(l, m) and the requisite kernelwidthpx.
"""
function mkgaussiantaper(gridspec, mu)
    maxr = (minimum([gridspec.Nx, gridspec.Ny]) ÷ 2 - 1) * gridspec.scalelm
    sigmalm = maxr * sqrt(-1 / log(mu))
    sigmauv = 1 / (2π * sigmalm)
    kernelwidthpx = round(Int, sqrt(-2 * sigmauv^2 * log(mu)) / gridspec.scaleuv)

    taper = (l, m) -> exp(-(l^2 + m^2) / (2 * sigmalm^2))
    return taper, kernelwidthpx, sigmalm
end

function mkkbtaper(gridspec, mu; alpha=14)
    maxr = (minimum([gridspec.Nx, gridspec.Ny]) ÷ 2 - 1) * gridspec.scalelm
    norm = besseli(0, π * alpha)

    function taper(l, m)
        r2 = (l^2 + m^2) / (2 * maxr)^2
        if r2 > 0.25
            return 0.
        else
            return besseli(0, π * alpha * sqrt(1 - 4 * r2)) / norm
        end
    end

    strip = zeros(gridspec.Nx)
    for lpx in 1:length(strip)
        l, m = px2sky(lpx, gridspec.Ny ÷ 2 + 1, gridspec)
        strip[lpx] = taper(l, m)
    end
    strip = abs.(fftshift(ifft(ifftshift(strip)))[gridspec.Nx ÷ 2 + 1:end])

    paddingpx = 0
    for (x, val) in enumerate(strip)
        println(val / strip[1])
        if val / strip[1] < mu
            paddingpx = x
            break
        end
    end

    return taper, paddingpx
end

