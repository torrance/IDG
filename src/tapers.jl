"""
    For a given gridspec, find the maximally sized Gaussian that reduces to threshold mu
    by the edge.

    Returns the taper function c(l, m) and the requisite kernelwidthpx.
"""
function mkgaussiantaper(gridspec, mu)
    maxr = (minimum([gridspec.Nx, gridspec.Ny]) ÷ 2 - 1) * gridspec.scalelm
    sigmalm = maxr * sqrt(-1 / log(mu))
    sigmauv = 1 / sigmalm
    kernelwidthpx = round(Int, sqrt(-2 * sigmauv^2 * log(mu)) / gridspec.scaleuv)
    println("sigmalm: $(sigmalm) sigmauv: $(sigmauv) kernelwidth: $(kernelwidthpx)")

    taper = (l, m) -> exp(-(l^2 + m^2) / (2 * sigmalm^2))
    return taper, kernelwidthpx
end

