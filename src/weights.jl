abstract type ImageWeight end
struct Natural <: ImageWeight
    normfactor::SVector{4, Float64}
end
struct Uniform <: ImageWeight end
struct Briggs <: ImageWeight
    imageweight::Array{Float64, 3}
    gridspec::GridSpec
    normfactor::SVector{4, Float64}
end

function Natural(uvdata, gridspec::GridSpec)
    normfactor = MVector{4, Float64}(0, 0, 0, 0)
    for uvdatum in uvdata
        upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if 1 <= upx <= gridspec.Nx && 1 <= vpx <= gridspec.Ny
            normfactor[1] += uvdatum.xx.re
            normfactor[2] += uvdatum.xy.re
            normfactor[3] += uvdatum.yx.re
            normfactor[4] += uvdatum.yy.re
        end
    end

    return Natural(normfactor)
end

function Briggs(uvdata, gridspec::GridSpec, robust::Float64)
    griddedweights = makegriddedweights(uvdata, gridspec)
    f2 = (5 * 10^-robust)^2 ./ (sum(x -> x^2, griddedweights, dims=(2, 3)) ./ sum(griddedweights, dims=(2, 3)))
    imageweights = 1 ./ (1 .+ griddedweights .* f2)

    println("Sum of weights: ", sum(griddedweights, dims=(2, 3)))
    normfactor = sum(imageweights .* griddedweights, dims=(2, 3))

    return Briggs(imageweights, gridspec, normfactor)
end

function makegriddedweights(uvdata, gridspec::GridSpec)
    griddedweights = zeros(4, gridspec.Nx, gridspec.Ny)

    for uvdatum in uvdata
        upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)

        if 1 <= upx <= gridspec.Nx && 1 <= vpx <= gridspec.Ny
            griddedweights[1, upx, vpx] += uvdatum.xx.re
            griddedweights[2, upx, vpx] += uvdatum.xy.re
            griddedweights[3, upx, vpx] += uvdatum.yx.re
            griddedweights[4, upx, vpx] += uvdatum.yy.re
        end
    end

    return griddedweights
end

@views function (w::Briggs)(ulambda, vlambda)
    upx, vpx = lambda2px(Int, ulambda, vlambda, w.gridspec)
    if upx in axes(w.imageweight, 2) && vpx in axes(w.imageweight, 3)
        return w.imageweight[:, upx, vpx]
    else
        return Float64[0, 0, 0, 0]
    end
end

function (w::Natural)(ulambda, vlambda)
    return SVector{4, Float32}(1, 1, 1, 1)
end