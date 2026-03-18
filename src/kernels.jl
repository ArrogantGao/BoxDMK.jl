const _NTAYLOR_3D = 2
const _FADDEEVA_SERIES_TERMS = 20

_erf(x::T) where {T<:Real} = T(ccall((:erf, Base.Math.libm), Cdouble, (Cdouble,), Float64(x)))

function _faddeevan_3d(ntaylor::Integer, a::T, delta::T) where {T<:Real}
    ntaylor <= 3 || throw(ArgumentError("3D Faddeeva moments are only implemented up to order 3"))

    x = a / sqrt(delta)
    x2 = x * x

    coeffs = Vector{T}(undef, ntaylor + 1)
    coeffs[1] = delta * sqrt(delta)
    for i in 2:length(coeffs)
        coeffs[i] = coeffs[i - 1] * delta
    end

    moments = zeros(T, ntaylor + 1)
    if x < one(T)
        x3 = x2 * x
        for k in 0:ntaylor
            x2k = x2^k
            di = x3
            fact = one(T)
            sign = one(T)
            acc = zero(T)

            for i in 0:_FADDEEVA_SERIES_TERMS
                acc += sign * di * x2k / (fact * (2 * i + 3 + 2 * k))
                di *= x2
                fact *= i + 1
                sign = -sign
            end

            moments[k + 1] = coeffs[k + 1] * acc
        end
        return moments
    end

    sqpi = sqrt(T(pi))
    erfx = _erf(x)
    expx2 = exp(-x2)
    x4 = x2 * x2
    x6 = x4 * x2

    moments[1] = coeffs[1] * (erfx * sqpi / 2 - x * expx2) / 2
    if ntaylor >= 1
        moments[2] = coeffs[2] * (
            3 * sqpi * erfx / 8 - x * expx2 * (x2 / 2 + T(0.75))
        )
    end
    if ntaylor >= 2
        moments[3] = coeffs[3] * (
            15 * sqpi * erfx / 16 - x * expx2 * (T(15) / 8 + 5 * x2 / 4 + x4 / 2)
        )
    end
    if ntaylor >= 3
        moments[4] = coeffs[4] * (
            105 * sqpi * erfx / 32 - x * expx2 * (T(105) / 16 + 35 * x2 / 8 + 7 * x4 / 4 + x6 / 2)
        )
    end

    return moments
end

function _subtract_gaussians!(coeffs::AbstractVector{T}, sog::SOGNodes{T}) where {T<:Real}
    length(sog.weights) == length(sog.deltas) || throw(ArgumentError("SOG weights and deltas must have the same length"))

    for (weight, delta) in zip(sog.weights, sog.deltas)
        moments = _faddeevan_3d(_NTAYLOR_3D, sog.r0, delta)
        @inbounds for k in eachindex(coeffs)
            coeffs[k] -= weight * moments[k]
        end
    end

    return coeffs
end

function _scale_taylor_coefficients_3d(coeffs::AbstractVector{T}) where {T<:Real}
    coeffs[1] *= 4 * T(pi)
    coeffs[2] *= 4 * T(pi) / 2 / 3
    coeffs[3] *= 4 * T(pi) / 24 / 5
    return coeffs
end

function _laplace_base_coefficients_3d(sog::SOGNodes{T}) where {T<:Real}
    r0 = sog.r0
    return T[
        r0^2 / 2,
        r0^4 / 4,
        r0^6 / 6,
    ]
end

function _sqrtlaplace_base_coefficients_3d(sog::SOGNodes{T}) where {T<:Real}
    r0 = sog.r0
    return T[
        r0,
        r0^3 / 3,
        r0^5 / 5,
    ]
end

function _yukawa_base_coefficients_3d(kernel::YukawaKernel, sog::SOGNodes{T}) where {T<:Real}
    beta = T(kernel.beta)
    iszero(beta) && return _laplace_base_coefficients_3d(sog)

    r0 = sog.r0
    br = beta * r0
    br2 = br * br
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    br7 = br6 * br
    br8 = br7 * br
    br9 = br8 * br

    b2 = beta * beta
    b4 = b2 * b2
    b6 = b2 * b4

    if abs(br) > T(1e-3)
        ebr = exp(-br)
        return T[
            (1 - ebr * (br + 1)) / b2,
            (6 - ebr * (br3 + 3 * br2 + 6 * br + 6)) / b4,
            (120 - ebr * (br5 + 5 * br4 + 20 * br3 + 60 * br2 + 120 * br + 120)) / b6,
        ]
    end

    return T[
        (br2 / 2 - br3 / 3 + br4 / 8 - br5 / 30 + br6 / 144) / b2,
        (br4 / 4 - br5 / 5 + br6 / 12 - br7 / 42) / b4,
        (br6 / 6 - br7 / 7 + br8 / 16 - br9 / 54) / b6,
    ]
end

function _taylor_coefficients_3d(coeffs::Vector{T}, sog::SOGNodes{T}) where {T<:Real}
    _subtract_gaussians!(coeffs, sog)
    _scale_taylor_coefficients_3d(coeffs)
    return (coeffs[1], coeffs[2], coeffs[3])
end

function taylor_coefficients(kernel::LaplaceKernel, sog::SOGNodes{T}) where {T<:Real}
    return _taylor_coefficients_3d(_laplace_base_coefficients_3d(sog), sog)
end

function taylor_coefficients(kernel::YukawaKernel, sog::SOGNodes{T}) where {T<:Real}
    return _taylor_coefficients_3d(_yukawa_base_coefficients_3d(kernel, sog), sog)
end

function taylor_coefficients(kernel::SqrtLaplaceKernel, sog::SOGNodes{T}) where {T<:Real}
    return _taylor_coefficients_3d(_sqrtlaplace_base_coefficients_3d(sog), sog)
end

function taylor_coefficients_grad(kernel::AbstractKernel, sog::SOGNodes)
    c0, c1, _ = taylor_coefficients(kernel, sog)
    return (c0, c1)
end

function taylor_coefficients_hess(kernel::AbstractKernel, sog::SOGNodes)
    c0, c1, _ = taylor_coefficients(kernel, sog)
    return (c0, c1)
end
