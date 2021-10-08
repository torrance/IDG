using StaticArrays

struct UVDatum
    u::Float32
    v::Float32
    w::Float32
    xx::ComplexF32
    xy::ComplexF32
    yx::ComplexF32
    yy::ComplexF32
end