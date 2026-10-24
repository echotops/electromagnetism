"""
jefimenko-feynman

This Julia script generates electric and magnetic fields using the
Jefimenko-Feynman (also known as Heaviside-Feynman) equations.

The result is four .jld2 files stored in ./cache that can be plotted
using plot.jl. There is one .jl2d file each for the charge positions,
electric fields, magnetic fields, and grid points.

I started this script at the end of highschool in early 2025
Since then I've added delayed time in late 2025, and optimization
in late 2026

echotops
10/23/2026
"""

# ------------------------------ Dependencies ------------------------------ #

using JLD2
using LinearAlgebra
using REPL
using StaticArrays

include("../calculus.jl")
include("../physics.jl")
include("../pc_configs.jl")

# ------------------------------ Parameters ------------------------------ #

const dt = 0.1 # Time step 
const steps = 200 # Number of steps
const xs = collect(LinRange(-5, 5, 15))
const ys = collect(LinRange(-5, 5, 15))
const zs = collect(LinRange(-5, 5, 15))
const grid = [SA[x, y, z] for x in xs for y in ys for z in zs]
const c_SI = 299792458.0 # For converting to SI
const c = 1 # For the simulation
const ϵ₀ = 8.8541878128e-12

# ------------------------------ Point Charges ------------------------------ #

function getPointCharges()
    # return [PointCharge(t -> SA[2 * cos(c * 0.4 * t), 0.0, 0.0], 1.0)]
    # return [PointCharge(t -> line_config(c * 0.1 * t, d), 1.0) for d in range(0, 1, 5)[begin:end-1]]
    # return [PointCharge(t -> loop_config(c * 0.5 * t, φ, r=2.5), 1.0) for φ in range(0, 2π, 8)[begin:end-1]]
    return [PointCharge(t -> toroid_config(c * 0.5 * t, φ), 1.0) for φ in range(0, 2π, 64)[begin:end-1]]
end

# ------------------------------ Simulation ------------------------------ #

function simulate()
    pointCharges = getPointCharges()
    runtimes = []
    chargePositions = []
    times = []
    electricFields = []
    magneticFields = []

    println("Running simulation with $steps steps and $dt timestep...")

    println()
    terminal = REPL.Terminals.TTYTerminal(string(), stdin, stdout, stderr)
    for t in 0:dt:(dt * steps)
        push!(runtimes, time())

        E, B = jefimenkoFeynman(pointCharges, t)

        push!(times, t * 1e9 / c_SI)
        push!(electricFields, E)
        push!(magneticFields, B)
        push!(chargePositions, [pc.x(t) for pc in pointCharges])

        text = "$(round(100 * t / (dt * steps), digits=3))%"
        REPL.Terminals.cmove_left(terminal, length(text))
        REPL.Terminals.cmove_line_up(terminal)
        println(text)
    end

    total_runtime = round(runtimes[end] - runtimes[begin], digits=3)
    runtime_per_step = round(total_runtime / steps, digits=3)
    println("Run completed in $total_runtime seconds, with $runtime_per_step seconds per step.")

    println("Saving run to the cache...")
    saveToBinary(times, grid, chargePositions, electricFields, magneticFields)
    println("Saved to the cache!")
end

function jefimenkoFeynman(pointCharges, t)
    function electromagneticField(x)
        e = zero(SVector{3,Float64})
        b = zero(SVector{3,Float64})
    
        for pc in pointCharges
            r̂(t) = normalize(pc.x(t) - x )
            r(t) = hypot((x - pc.x(t))... )

            t′ = tprime(x, pc.x, t, c)
            r̂_t′ = r̂(t′)
            r_t′ = r(t′)
            g(τ) = 1 / (1 + dot(r̂(τ), d_dt(pc.x, τ)) / c)

            e_1 = r̂_t′ / r_t′^2
            e_2 = r_t′ * d_dt(t -> (r̂(t) / r(t)^2), t′) * g(t′) / c
            e_3 = d_dt(t -> d_dt(r̂, t) * g(t), t′) * g(t′) / c^2
            e_q = -pc.q * (e_1 + e_2 + e_3) / (4 * π * ϵ₀)

            e += e_q 
            b += cross(-r̂_t′, e_q / c)
        end

        return e, b
    end

    E = Vector{SVector{3,Float64}}(undef, length(grid))
    B = Vector{SVector{3,Float64}}(undef, length(grid))

    Threads.@threads for i in eachindex(grid)
        e, b = electromagneticField(grid[i])

        E[i] = e
        B[i] = b
    end

    return E, B
end

function saveToBinary(times, grid, chargePositions, electricFields, magneticFields)
    jldsave("3D/cache/times.jld2"; times=times)
    jldsave("3D/cache/grid.jld2"; grid=grid)
    jldsave("3D/cache/charges.jld2"; charges=chargePositions)
    jldsave("3D/cache/electric.jld2"; E=electricFields)
    jldsave("3D/cache/magnetic.jld2"; B=magneticFields)
end

simulate()