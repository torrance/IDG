import Base: convert, *

struct ComplexAP{T} <: Number
    amplitude::T
    phase::T
end

const ComplexAP32 = ComplexAP{Float32}
const ComplexAP64 = ComplexAP{Float64}

convert(::Type{Complex{T}}, x::ComplexAP{P}) where {T <: Real, P <: Real} = x.amplitude * exp(1im * x.phase)
convert(::Type{ComplexAP{T}}, x::Complex{P}) where {T <: Real, P <: Real} = ComplexAP(hypot(x.re, x.im), atan(x.im, x.re))

(*)(x::ComplexAP{T}, y::ComplexAP{P}) where {T <: Real, P <: Real} = ComplexAP(x.amplitude * y.amplitude, x.phase + y.phase)

