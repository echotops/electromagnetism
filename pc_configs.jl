"""
pc-configs

A few interesting configurations of point charges

echotops
10/23/2026
"""

using StaticArrays

function line_config(t, d)
    s = t + d
    return SA[s, 0.0, 0.0]
end

function loop_config(t, φ; r=1)
    speed = r
    s = t / speed + φ
    return SA[r * cos(s), r * sin(s), 0.0]
end

function toroid_config(t, φ; windings=16, r=0.5, R=2.5)
    speed = hypot(abs(R) + abs(r), r * windings)
    s = t / speed + φ
    return SA[cos(s) * (R + r * sin(windings * s)), sin(s) * (R + r * sin(windings * s)), r * cos(windings * s)]
end