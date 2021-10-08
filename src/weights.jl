abstract type ImageWeight end
struct Natural <: ImageWeight end
struct Uniform <: ImageWeight end
struct Briggs <: ImageWeight
    imageweight::Array{Float64, 3}
    gridspec::GridSpec
end

function Briggs(mset::MeasurementSet, gridspec::GridSpec, robust::Float64)
    gridcount = makegridcount(mset, gridspec)
    f2 = (5 * 10^-robust)^2 ./ (sum(x -> x^2, gridcount, dims=(2, 3)) ./ sum(gridcount, dims=(2, 3)))
    
    return Briggs(1 ./ (1 .+ gridcount .* f2), gridspec)
end

function makegridcount(mset::MeasurementSet, gridspec::GridSpec)
    npol = 4  # Hardcode 4 polarizations for now
    gridcount = zeros(npol, gridspec.Ny, gridspec.Ny)

    @views for row in mset
        makegridcount!(gridcount, row, mset.lambdas, gridspec)
    end
    return gridcount
end

function makegridcount!(gridcount, row, lambdas, gridspec)
    for (chan, lambda) in enumerate(lambdas)
        upx, vpx = lambda2px(Int, row.uvw[1] / lambda, row.uvw[2] / lambda, gridspec)

        if upx in axes(gridcount, 2) && vpx in axes(gridcount, 3)
            for pol in 1:4
                @inbounds gridcount[pol, upx, vpx] += (
                    !row.flag[pol, chan] *
                    !row.flagrow[] * 
                    row.weight[pol] *
                    row.weightspectrum[pol, chan]
                )
            end
        end
    end
end

@views function (w::Briggs)(ulambda, vlambda)
    upx, vpx = lambda2px(Int, ulambda, vlambda, w.gridspec)
    if upx in axes(w.imageweight, 2) && vpx in axes(w.imageweight, 3)
        return w.imageweight[:, upx, vpx]
    else
        return Float64[0, 0, 0, 0]
    end
end

