"""
physics

This Julia script contains structs and functions used in both the causal-rk4
and jefimenk-feynman simulations.

echotops
10/23/2026
"""

struct PointCharge{F<:Function}
    x::F
    q::Float64
end

function tprime(point, charge_position, t, c, tol=1e-10)
    Δt_low = 0
    Δt_high = hypot((charge_position(t) - point)...) / c
    residual(Δt) = hypot((charge_position(t - Δt) - point)...) - c * Δt
    while residual(Δt_high) > 0
        Δt_high *= 2
    end
    res = Inf
    Δt = 0

    for _ in 1:100
        Δt = (Δt_low + Δt_high) / 2
        res = residual(Δt)
        if res > 0
            Δt_low = Δt
        else
            Δt_high = Δt
        end

        if abs(res) < tol
            break
        end
    end

    return t - Δt
end