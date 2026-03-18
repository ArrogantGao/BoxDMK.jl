function _check_basis_order(n::Integer)
    n > 0 || throw(ArgumentError("basis order n must be positive"))
    return Int(n)
end

function _legendre_rule(n::Int)
    if n == 1
        return [0.0], [2.0]
    end

    beta = [k / sqrt(4 * k^2 - 1) for k in 1:(n - 1)]
    jacobi = SymTridiagonal(zeros(n), beta)
    decomposition = eigen(jacobi)

    return setprecision(BigFloat, 256) do
        roots = BigFloat.(decomposition.values)
        weights = Vector{BigFloat}(undef, n)

        for i in eachindex(roots)
            root = roots[i]

            for _ in 1:6
                polynomial, derivative = _legendre_polynomial_and_derivative(root, n)
                delta = polynomial / derivative
                root -= delta
                abs(delta) <= eps(root) && break
            end

            _, derivative = _legendre_polynomial_and_derivative(root, n)
            roots[i] = root
            weights[i] = 2 / ((1 - root^2) * derivative^2)
        end

        for i in 1:(n ÷ 2)
            j = n - i + 1
            root = (abs(roots[i]) + abs(roots[j])) / 2
            weight = (weights[i] + weights[j]) / 2
            roots[i] = -root
            roots[j] = root
            weights[i] = weight
            weights[j] = weight
        end

        if isodd(n)
            roots[(n + 1) ÷ 2] = zero(BigFloat)
        end

        return Float64.(roots), Float64.(weights)
    end
end

function _legendre_eval_matrices(points::AbstractVector, n::Int)
    m = length(points)
    T = float(promote_type(eltype(points), Float64))

    values = Matrix{T}(undef, m, n)
    first_derivatives = Matrix{T}(undef, m, n)
    second_derivatives = Matrix{T}(undef, m, n)

    for i in 1:m
        x = T(points[i])

        values[i, 1] = one(T)
        first_derivatives[i, 1] = zero(T)
        second_derivatives[i, 1] = zero(T)

        if n == 1
            continue
        end

        values[i, 2] = x
        first_derivatives[i, 2] = one(T)
        second_derivatives[i, 2] = zero(T)

        pjm2 = one(T)
        pjm1 = x
        derjm1 = one(T)
        der2jm1 = zero(T)

        for j in 2:(n - 1)
            pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j
            derj = j * pjm1 + x * derjm1
            der2j = (j + 1) * derjm1 + x * der2jm1

            values[i, j + 1] = pj
            first_derivatives[i, j + 1] = derj
            second_derivatives[i, j + 1] = der2j

            pjm2 = pjm1
            pjm1 = pj
            derjm1 = derj
            der2jm1 = der2j
        end
    end

    return values, first_derivatives, second_derivatives
end

function _legendre_mode_scaling(n::Int)
    return [(2 * k + 1) / 2 for k in 0:(n - 1)]
end

function _legendre_polynomial_and_derivative(x::Real, n::Int)
    if n == 0
        return one(x), zero(x)
    end

    pkm1 = one(x)
    pk = x

    if n == 1
        return pk, one(x)
    end

    for k in 1:(n - 1)
        pkm1, pk = pk, ((2 * k + 1) * x * pk - k * pkm1) / (k + 1)
    end

    derivative = n * (x * pk - pkm1) / (x^2 - 1)
    return pk, derivative
end

function _legendre_barycentric_weights(x::AbstractVector, w::AbstractVector)
    T = promote_type(eltype(x), eltype(w))
    barycentric = Vector{T}(undef, length(x))

    for i in eachindex(x)
        sign = isodd(i) ? -one(T) : one(T)
        barycentric[i] = sign * sqrt((1 - x[i]^2) * w[i])
    end

    return barycentric
end

function _legendre_first_derivative_matrix(x::AbstractVector, w::AbstractVector)
    n = length(x)
    T = promote_type(eltype(x), eltype(w))
    derivative = zeros(T, n, n)

    n == 1 && return derivative

    barycentric = _legendre_barycentric_weights(x, w)

    for i in 1:n
        rowsum = zero(T)

        for j in 1:n
            i == j && continue

            entry = barycentric[j] / (barycentric[i] * (x[i] - x[j]))
            derivative[i, j] = entry
            rowsum += entry
        end

        derivative[i, i] = -rowsum
    end

    return derivative
end

function _legendre_second_derivative_matrix(x::AbstractVector, w::AbstractVector)
    n = length(x)
    T = promote_type(eltype(x), eltype(w))
    second_derivative = zeros(T, n, n)

    n == 1 && return second_derivative

    derivative = _legendre_first_derivative_matrix(x, w)

    for i in 1:n
        rowsum = zero(T)

        for j in 1:n
            i == j && continue

            entry = 2 * derivative[i, j] * (derivative[i, i] - inv(x[i] - x[j]))
            second_derivative[i, j] = entry
            rowsum += entry
        end

        second_derivative[i, i] = -rowsum
    end

    return second_derivative
end

function _legendre_gauss_data(n::Int)
    x, w = _legendre_rule(n)
    values, first_derivatives, second_derivatives = _legendre_eval_matrices(x, n)
    inverse = values
    forward = Diagonal(_legendre_mode_scaling(n)) * transpose(values) * Diagonal(w)
    return x, w, forward, inverse, first_derivatives, second_derivatives
end

function nodes_and_weights(::LegendreBasis, n::Integer)
    order = _check_basis_order(n)
    return _legendre_rule(order)
end

function forward_transform(::LegendreBasis, n::Integer)
    order = _check_basis_order(n)
    _, _, forward, _, _, _ = _legendre_gauss_data(order)
    return forward
end

function inverse_transform(::LegendreBasis, n::Integer)
    order = _check_basis_order(n)
    _, _, _, inverse, _, _ = _legendre_gauss_data(order)
    return inverse
end

function derivative_matrix(::LegendreBasis, n::Integer)
    order = _check_basis_order(n)
    x, w = _legendre_rule(order)
    return setprecision(BigFloat, 256) do
        Float64.(_legendre_first_derivative_matrix(BigFloat.(x), BigFloat.(w)))
    end
end

function second_derivative_matrix(::LegendreBasis, n::Integer)
    order = _check_basis_order(n)
    x, w = _legendre_rule(order)
    return setprecision(BigFloat, 256) do
        Float64.(_legendre_second_derivative_matrix(BigFloat.(x), BigFloat.(w)))
    end
end

function interpolation_matrix(::LegendreBasis, from_nodes::AbstractVector, to_nodes::AbstractVector)
    n = _check_basis_order(length(from_nodes))
    length(to_nodes) == 0 && return Matrix{Float64}(undef, 0, n)

    values_from, _, _ = _legendre_eval_matrices(from_nodes, n)
    values_to, _, _ = _legendre_eval_matrices(to_nodes, n)
    return values_to / values_from
end

function _chebyshev_nodes(::Type{T}, n::Int) where {T<:AbstractFloat}
    nodes = Vector{T}(undef, n)
    factor = T(pi) / (2 * T(n))

    for i in 1:n
        nodes[n - i + 1] = cos(T(2 * i - 1) * factor)
    end

    return nodes
end

function _chebyshev_eval_matrices(points::AbstractVector, n::Int)
    m = length(points)
    T = float(promote_type(eltype(points), Float64))

    values = Matrix{T}(undef, m, n)
    first_derivatives = Matrix{T}(undef, m, n)
    second_derivatives = Matrix{T}(undef, m, n)

    for i in 1:m
        x = T(points[i])

        values[i, 1] = one(T)
        first_derivatives[i, 1] = zero(T)
        second_derivatives[i, 1] = zero(T)

        if n == 1
            continue
        end

        values[i, 2] = x
        first_derivatives[i, 2] = one(T)
        second_derivatives[i, 2] = zero(T)

        tjm2 = one(T)
        tjm1 = x
        derjm2 = zero(T)
        derjm1 = one(T)
        der2jm2 = zero(T)
        der2jm1 = zero(T)

        for j in 2:(n - 1)
            tj = 2 * x * tjm1 - tjm2
            derj = 2 * tjm1 + 2 * x * derjm1 - derjm2
            der2j = 4 * derjm1 + 2 * x * der2jm1 - der2jm2

            values[i, j + 1] = tj
            first_derivatives[i, j + 1] = derj
            second_derivatives[i, j + 1] = der2j

            tjm2 = tjm1
            tjm1 = tj
            derjm2 = derjm1
            derjm1 = derj
            der2jm2 = der2jm1
            der2jm1 = der2j
        end
    end

    return values, first_derivatives, second_derivatives
end

function _chebyshev_mode_scaling(::Type{T}, n::Int) where {T<:AbstractFloat}
    scaling = fill(T(2) / T(n), n)
    scaling[1] = inv(T(n))
    return scaling
end

function _chebyshev_moments(::Type{T}, n::Int) where {T<:AbstractFloat}
    return [iseven(k) ? T(2) / (one(T) - T(k)^2) : zero(T) for k in 0:(n - 1)]
end

function _chebyshev_gauss_data(::Type{T}, n::Int) where {T<:AbstractFloat}
    x = _chebyshev_nodes(T, n)
    values, first_derivatives, second_derivatives = _chebyshev_eval_matrices(x, n)
    inverse = values
    forward = Diagonal(_chebyshev_mode_scaling(T, n)) * transpose(values)
    weights = transpose(forward) * _chebyshev_moments(T, n)
    return x, weights, forward, inverse, first_derivatives, second_derivatives
end

function nodes_and_weights(::ChebyshevBasis, n::Integer)
    order = _check_basis_order(n)
    return setprecision(BigFloat, 256) do
        x, w, _, _, _, _ = _chebyshev_gauss_data(BigFloat, order)
        return Float64.(x), Float64.(w)
    end
end

function forward_transform(::ChebyshevBasis, n::Integer)
    order = _check_basis_order(n)
    return setprecision(BigFloat, 256) do
        _, _, forward, _, _, _ = _chebyshev_gauss_data(BigFloat, order)
        return Float64.(forward)
    end
end

function inverse_transform(::ChebyshevBasis, n::Integer)
    order = _check_basis_order(n)
    return setprecision(BigFloat, 256) do
        _, _, _, inverse, _, _ = _chebyshev_gauss_data(BigFloat, order)
        return Float64.(inverse)
    end
end

function derivative_matrix(::ChebyshevBasis, n::Integer)
    order = _check_basis_order(n)
    return setprecision(BigFloat, 256) do
        _, _, forward, _, first_derivatives, _ = _chebyshev_gauss_data(BigFloat, order)
        return Float64.(first_derivatives * forward)
    end
end

function second_derivative_matrix(::ChebyshevBasis, n::Integer)
    order = _check_basis_order(n)
    return setprecision(BigFloat, 256) do
        _, _, forward, _, _, second_derivatives = _chebyshev_gauss_data(BigFloat, order)
        return Float64.(second_derivatives * forward)
    end
end

function interpolation_matrix(::ChebyshevBasis, from_nodes::AbstractVector, to_nodes::AbstractVector)
    n = _check_basis_order(length(from_nodes))
    length(to_nodes) == 0 && return Matrix{Float64}(undef, 0, n)

    values_from, _, _ = _chebyshev_eval_matrices(from_nodes, n)
    values_to, _, _ = _chebyshev_eval_matrices(to_nodes, n)
    return values_to / values_from
end
