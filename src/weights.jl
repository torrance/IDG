abstract type ImageWeight end
struct Natural <: ImageWeight
    totalweights::SVector{4, Float64}
end
struct Uniform <: ImageWeight end
struct Briggs <: ImageWeight
    imageweight::Array{Float64, 3}
    gridspec::GridSpec
    normfactor::SVector{4, Float64}
end

function Natural(mset::MeasurementSet, gridspec::GridSpec)
    totalweights = Float64[0, 0, 0, 0]
    @views for row in mset
        sumweights!(totalweights, row, mset.lambdas, gridspec)
    end
    return Natural(totalweights)
end

function sumweights!(totalweights, row, lambdas, gridspec)
    for (chan, lambda) in enumerate(lambdas)
        upx, vpx = lambda2px(Int, row.uvw[1] / lambda, row.uvw[2] / lambda, gridspec)

        if 1 <= upx <= gridspec.Nx && 1 <= vpx <= gridspec.Ny
            for pol in 1:4
                weight = (
                    !row.flag[pol, chan] *
                    !row.flagrow[] * 
                    row.weight[pol] *
                    row.weightspectrum[pol, chan]
                )
                
                if isfinite(weight)
                    totalweights[pol] += weight
                end
            end
        end
    end
end

function Briggs(mset::MeasurementSet, gridspec::GridSpec, robust::Float64)
    griddedweights = makegriddedweights(mset, gridspec)
    f2 = (5 * 10^-robust)^2 ./ (sum(x -> x^2, griddedweights, dims=(2, 3)) ./ sum(griddedweights, dims=(2, 3)))
    imageweights = 1 ./ (1 .+ griddedweights .* f2)

    println("Sum of weights: ", sum(griddedweights, dims=(2, 3)))
    normfactor = sum(imageweights .* griddedweights, dims=(2, 3))

    return Briggs(imageweights, gridspec, normfactor)
end

function makegriddedweights(mset::MeasurementSet, gridspec::GridSpec)
    npol = 4  # Hardcode 4 polarizations for now
    griddedweights = zeros(npol, gridspec.Nx, gridspec.Ny)
    @views for row in mset
        makegriddedweights!(griddedweights, row, mset.lambdas, gridspec)
    end

    return griddedweights
end

function makegriddedweights!(griddedweights, row, lambdas, gridspec)
    for (chan, lambda) in enumerate(lambdas)
        upx, vpx = lambda2px(Int, row.uvw[1] / lambda, -row.uvw[2] / lambda, gridspec)

        if upx in axes(griddedweights, 2) && vpx in axes(griddedweights, 3)
            for pol in 1:4
                weight = (
                    !row.flag[pol, chan] *
                    !row.flagrow[] * 
                    row.weight[pol] *
                    row.weightspectrum[pol, chan]
                )
                
                if isfinite(weight)
                    griddedweights[pol, upx, vpx] += weight
                end
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

function (w::Natural)(ulambda, vlambda)
    return 1 / w.totalweights
end