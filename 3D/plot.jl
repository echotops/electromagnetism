"""
plot

This Julia script generates a plot and .mp4 video of the electric and magnetic
fields generated using either the causal-rk4 or jefimenk-feynman scripts.

echotops
10/23/2026
"""

# ------------------------------ Dependencies ------------------------------ #

using GLMakie
using JLD2
using REPL
using StaticArrays
using Statistics

# ------------------------------ Parameters ------------------------------ #

const framerate = 30
const c_SI = 299792458.0 # For converting to SI

# ------------------------------ Plotting ------------------------------ #

function plot(filename, figure_size, limits;
    electric_field=true, magnetic_field=true,
    electric_limits=(0.95, 0.5), magnetic_limits=(0.95, 0.5),
    electric_colorbar=false, magnetic_colorbar=false)

    set_theme!(merge(theme_black(), theme_latexfonts()))
    fig = Figure(size=figure_size)
    ax = Axis3(fig[1, 1], limits=limits, aspect=:data)
    hidespines!(ax)

    println("Loading run from the cache...")
    times, grid, charges, Es, Bs = loadFromBinary()
    println("Loaded from the cache!")

    points = [Point3f(g) for g in grid]

    println("Gathering statistics...")

    Es_lengths = [[hypot(e...) for e in E] for E in Es]
    Es_all_lengths = reduce(vcat, Es_lengths)
    filter!(isfinite, Es_all_lengths)
    electric_scale = quantile(Es_all_lengths, electric_limits[1])
    electric_cutoff = quantile(Es_all_lengths, electric_limits[2])
    Bs_lengths = [[hypot(b...) for b in B] for B in Bs]
    Bs_all_lengths = reduce(vcat, Bs_lengths)
    filter!(isfinite, Bs_all_lengths)
    magnetic_scale = quantile(Bs_all_lengths, magnetic_limits[1]) 
    magnetic_cutoff = quantile(Bs_all_lengths, magnetic_limits[2])

    show_electric_colorbar = electric_field && (electric_limits[1] > electric_limits[2]) && electric_colorbar
    show_magnetic_colorbar = magnetic_field && (magnetic_limits[1] > magnetic_limits[2]) && magnetic_colorbar

    if show_electric_colorbar
        electric_colormap = [electric_color(l, electric_cutoff, electric_scale) for l in range(electric_cutoff, electric_scale, length=256)]
        Colorbar(fig[3, 1], limits=(electric_cutoff, electric_scale), colormap=electric_colormap, label="|E| (N/C)", vertical=false, flipaxis=false, labelsize=20, ticklabelsize=20)
    end

    if show_magnetic_colorbar
        row = show_electric_colorbar ? 4 : 3
        magnetic_colormap = [magnetic_color(l, magnetic_cutoff, magnetic_scale) for l in range(magnetic_cutoff, magnetic_scale, length=256)]
        Colorbar(fig[row, 1], limits=(magnetic_cutoff / c_SI, magnetic_scale / c_SI), colormap=magnetic_colormap, label="|B| (T)", vertical=false, flipaxis=false, labelsize=20, ticklabelsize=20)
    end

    println("Plotting simulation...")

    time_text = Observable("t = 0.00 ns")
    Label(fig[2, 1], time_text, halign=:left, tellwidth=false, fontsize=20)

    charge_obs = Observable(Point3f.(charges[1]))
    scatter!(ax, charge_obs, color=:white, markersize=8)
    arrow_plots = []

    println()
    terminal = REPL.Terminals.TTYTerminal(string(), stdin, stdout, stderr)
    record(fig, filename * ".mp4", 1:length(Es); framerate = framerate) do i
        foreach(p -> delete!(ax, p), arrow_plots); empty!(arrow_plots)

        time_text[] = "t = $(round(times[i], digits=2)) ns"

        E = Es[i]
        B = Bs[i]
        charge_positions = charges[i]

        electric_lengths = Es_lengths[i]
        electric_colors = [electric_color(l, electric_cutoff, electric_scale) for l in electric_lengths]
        electric_keep = findall(>=(electric_cutoff), electric_lengths)

        magnetic_lengths = Bs_lengths[i]
        magnetic_colors = [magnetic_color(l, magnetic_cutoff, magnetic_scale) for l in magnetic_lengths]
        magnetic_keep = findall(>=(magnetic_cutoff), magnetic_lengths)

        if electric_field
            push!(arrow_plots,
                arrows3d!(
                    ax,points[electric_keep], [Vec3f(e) for e in E[electric_keep]], normalize=true,
                    color=electric_colors[electric_keep], lengthscale=0.5, tipradius=0.075, tiplength=0.15,
                    shaftradius=0.02, transparency=true, align=:center
                )
            )
        end

        if magnetic_field
            push!(arrow_plots,
                arrows3d!(
                    ax, points[magnetic_keep], [Vec3f(b) for b in B[magnetic_keep]], normalize=true,
                    color=magnetic_colors[magnetic_keep], lengthscale=0.5, tipradius=0.075, tiplength=0.15,
                    shaftradius=0.02, transparency=true, align=:center
                )
            )
        end
        
        charge_obs[] = Point3f.(charge_positions)

        if (i % 10 == 0)
            text = "Frame $i of $(length(Es))"
            REPL.Terminals.cmove_left(terminal, length(text))
            REPL.Terminals.cmove_line_up(terminal)
            println(text)
        end
        
        ax.elevation[] = π/8 + 3π/8 * smooth_bump_pan(i / length(Es), 0, 0.25)
        ax.azimuth[] = 3π/8 + π/8 * smooth_bump_pan(i / length(Es), 0, 0.25)
    end

    println("Plot complete! Saved to \" " * filename * ".mp4\" ")
end

function loadFromBinary()
    times = load("3D/cache/times.jld2", "times")
    grid = load("3D/cache/grid.jld2", "grid")
    charges = load("3D/cache/charges.jld2", "charges")
    Es = load("3D/cache/electric.jld2", "E")
    Bs = load("3D/cache/magnetic.jld2", "B")

    return times, grid, charges, Es, Bs
end

function smooth_pan(t)
    f(t) = 3t^2 - 2t^3
    if t < 0
        return 0
    elseif t < 1
        return f(t)
    end
    return 1
end

function smooth_bump_pan(t, s=0.5, d=0)
    f(t) = 3t^2 - 2t^3
    g(t) = f(t / (s - d / 2)) * s
    h(t) = f(-(t - 1) / (1 - s - d / 2)) * (s - 1) + 1
    if t < 0
        return 0
    elseif t <= (s - d / 2)
        return g(t)
    elseif t <= (s + d / 2)
        return g(s - d / 2)
    elseif t <= 1
        return h(t)
    end
    return 1
end

function magnetic_color(length, cutoff, scale)
    s = clamp((length - cutoff) / (scale - cutoff), 0, 1)
    return RGBAf(0, s, min(1, 2 * s), s)
end

function electric_color(length, cutoff, scale)
    s = clamp((length - cutoff) / (scale - cutoff), 0, 1)
    return RGBAf(1, 0, 0, s)
end

plot("sim_run", (1000, 1000), (-4, 4, -4, 4, -4, 4),
    electric_field=false, magnetic_limits=(0.98, 0.9), magnetic_colorbar=false)