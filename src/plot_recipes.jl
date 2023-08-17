#using RecipesBase
using ColorTypes
#using MakieCore
using LaTeXStrings
using CairoMakie
import CairoMakie.GeometryBasics: Point2f

export color_palette
export draw_ellipse!
export data_histogram, show_singularities, show_singularities!, show_singularity_samples,
        show_singularity_samples!, plot_expected_sourcefield, plot_expected_sourcefield!, singularity_ellipses,
        singularity_ellipses!, plot_temperature_field, plot_temperature_field!,
        plot_sensor_data, plot_sensor_data!, plot_sensors!, draw_ellipse_x!, 
        draw_ellipse_y!, draw_ellipse_z!, get_ellipse_coords, show_sampling_history!

"""
A palette of colors for plotting
"""
const color_palette = [colorant"firebrick";
                       colorant"seagreen4";
                       colorant"goldenrod1";
                       colorant"skyblue1";
                       colorant"slateblue";
                       colorant"maroon3";
                       colorant"orangered2";
                       colorant"grey70";
                       colorant"dodgerblue4"]

#const color_palette2 = cgrad(:tab20, 10, categorical = true)

"""
    data_histogram(x::Vector[;bins=80,xlims=(-2,2)])

Create a histogram of the data in vector `x`.
"""
function data_histogram(x::Vector{T};bins=80,xlims = (-2,2), kwargs...) where {T<:Real}
    f = Figure()
    ax1 = f[1, 1] = Axis(f;kwargs...)
    hist!(ax1,x,bins=bins)
    xlims!(ax1, xlims...)
    f
end


function show_singularities!(ax,x::Vector,obs::AbstractObservationOperator;kwargs...)
    sing_array = state_to_singularity_states(x,obs.config)
    bluered = range(colorant"darkorange1",colorant"cornflowerblue",length=2)
    colormap = cgrad(bluered,0.25,categorical=true)
    sgns = sign.(sing_array[3,:])
    scatter!(ax,sing_array[1,:],sing_array[2,:];colormap=colormap,color=sgns,kwargs...)
end


"""
    show_singularities(x::Vector,obs::AbstractObservationOperator)

Given state vector `x`, create a plot that depicts the singularity positions.
"""
function show_singularities(x::Vector,obs::AbstractObservationOperator;kwargs...)
    f = Figure()
    ax = f[1, 1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y",kwargs...)
    show_singularities!(ax,x,obs)
    f
end


function show_singularity_samples!(ax,x_samples::Union{Array,BasicEnsembleMatrix},obs::AbstractObservationOperator;nskip=1,kwargs...)
    sing_array = states_to_singularity_states(x_samples[:,1:nskip:end],obs.config)
    sgns = sign.(sing_array[3,:])
    bluered = range(colorant"lightsalmon",colorant"lightskyblue1",length=2)
    colormap = cgrad(bluered,0.25,categorical=true)

    scatter!(ax,sing_array[1,:],sing_array[2,:];markersize=1.5,colormap=colormap,color=sgns,label="sample",kwargs...)

end


"""
    show_singularity_samples(x_samples,obs[;nskip=1])

Plot the samples of an ensemble of states as a scatter plot of singularity positions. Note that
`x_samples` must be of size Nstate x Ne. Use `nskip` to plot every `nskip` state.
"""
function show_singularity_samples(x_samples::Array,obs::AbstractObservationOperator;nskip=1,xlims=(-2,2),ylims=(-2,2),kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    show_singularity_samples!(ax,x_samples,obs;nskip=nskip,kwargs...)
    xlims!(ax,xlims...)
    ylims!(ax,ylims...)
    f
end

function show_sampling_history!(ax,i,x_samples::Union{Array,BasicEnsembleMatrix};nskip=1,kwargs...)
    sample_idx = 1:size(x_samples)[2]
    scatter!(ax,sample_idx,x_samples[i,:];markersize=5,label="sample",kwargs...)
end

function show_sampling_history(i,x_samples::Union{Array,BasicEnsembleMatrix};nskip=1,ylims=(-2,2),kwargs...)
    sample_idx = 1:size(x_samples)[2]
    f = Figure()
    ax = f[1,1] = Axis(f;limits=(0,length(sample_idx),ylims...),xlabel=L"iterations")
    scatter!(ax,sample_idx,x_samples[i,:];markersize=5,label="sample",kwargs...)
    f
end


function plot_expected_sourcefield!(ax,μ::Vector,Σ,obs::AbstractObservationOperator;xlims = (-2.5,2.5),Nx = 201, ylims = (-2.5,2.5), Ny = 201,kwargs...)
    xg = range(xlims...,length=Nx)
    yg = range(ylims...,length=Ny)
    h = [heaterfield(x,y,μ,Σ,obs.config) for x in xg, y in yg]
    contour!(ax,xg,yg,h;kwargs...)
    xlims!(ax,xlims...)
    ylims!(ax,ylims...)
end

function plot_expected_sourcefield!(ax,μ::AbstractMatrix,Σ,wts,obs::AbstractObservationOperator;xlims = (-2.5,2.5),Nx = 201, ylims = (-2.5,2.5), Ny = 201,kwargs...)
    xg = range(xlims...,length=Nx)
    yg = range(ylims...,length=Ny)
    h = zeros(Nx,Ny)
    for c in 1:size(μ,2)
      h .+= [wts[c]*heaterfield(x,y,μ[:,c],Σ[c],obs.config) for x in xg, y in yg]
    end
    contour!(ax,xg,yg,h;kwargs...)
    xlims!(ax,xlims...)
    ylims!(ax,ylims...)
end


"""
        plot_expected_sourcefield(μ,Σ,obs::AbstractObservationOperator[;xlims=(-2.5,2.5),Nx = 201,ylims=(-2.5,2.5),Ny = 201])

For a given mean state `μ` and state covariance `Σ`, calculate the expected value of the heater
field on a grid. The optional arguments allow one to specify the dimensions of the grid.
"""
function plot_expected_sourcefield(μ,Σ,obs::AbstractObservationOperator; kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    plot_expected_sourcefield!(ax,μ,Σ,obs;kwargs...)
    f
end


function singularity_ellipses!(ax,μ::Vector{T},Σ,obs::AbstractObservationOperator; kwargs...) where {T<:Real}
    N = number_of_singularities(obs.config)
    for j = 1:N
        xidj, yidj, qidj, c1idj, c1idj = get_singularity_ids(j,obs.config)
        μxj = μ[[xidj,yidj]]
        Σxxj = Σ[xidj:yidj,xidj:yidj]
        draw_ellipse!(ax,μxj,Σxxj;kwargs...)
    end
end


function singularity_ellipses!(ax,μ::AbstractMatrix{T},Σ,wts,obs::AbstractObservationOperator; threshold = 0.05, kwargs...) where {T<:Real}
    id_sort = sortperm(wts,rev=true)
    for c in 1:size(μ,2)
        id = id_sort[c]
        wts[id] < threshold && continue
        singularity_ellipses!(ax,μ[:,id],Σ[id],obs;color=Cycled(c),linewidth=1.5wts[id]^2/maximum(wts.^2),kwargs...)
    end
  end


  """
  singularity_ellipses(μ,Σ,obs::AbstractObservationOperator)

Given state mean `μ` and state covariance `Σ`, plot ellipses of uncertainty at each of the
mean singularity locations in `μ`.
"""
function singularity_ellipses(μ,Σ,obs::AbstractObservationOperator; kwargs...)
  f = Figure()
  ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
  singularity_ellipses!(ax,μ,Σ,obs;kwargs...)
  f
end

function plot_temperature_field!(ax,x::Vector,obs::TemperatureObservations,gridConfig::constructGrids;kwargs...)
    T = analytical_temperature(x,obs,gridConfig,true)
    @unpack g = gridConfig
    xg, yg = coordinates(T,g)
    plot_temperature_field!(ax,collect(xg),collect(yg),Matrix(T),obs;kwargs...)
end

"""
        plot_temperature_field(x::Vector,obs::TemperatureObservations,gridConfig::constructGrids[;xlims=(-2.5,2.5),Nx = 201,ylims=(-2.5,2.5),Ny = 201])

For a given state `x`, calculate the pressure field on a grid. The optional arguments allow one to specify the dimensions of the grid.
"""
function plot_temperature_field(x::Vector,obs::TemperatureObservations,gridConfig::constructGrids; kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    plot_temperature_field!(ax,x,obs,gridConfig;kwargs...)
    f
end

function plot_temperature_field!(ax,xg::AbstractVector,yg::AbstractVector,T::Matrix,obs::AbstractObservationOperator; levels = range(-0.5,0.01,length=21), kwargs...)
    contour!(ax,xg,yg,T;levels=levels, kwargs...)
    #plot_sensors!(ax,obs)
end


function plot_sensor_data!(ax,ystar::Vector,x::Vector,t::Real,obs::AbstractObservationOperator,gridConfig::constructGrids; sensor_noise=zero(ystar))
    plot_sensor_data!(ax,ystar,obs;sensor_noise=sensor_noise)
    y_est = observations(x,t,obs,gridConfig)
    scatter!(ax,y_est,markersize=15,color=:transparent,strokewidth=1,label="estimate")
end

function plot_sensor_data!(ax,ystar::Vector,obs::AbstractObservationOperator; sensor_noise=zero(ystar))
    scatter!(ax,ystar,markersize=10,color=:black,label="truth")
    errorbars!(ax,1:length(ystar),ystar,sensor_noise)
end


"""
    plot_sensor_data(ystar::Vector,obs::AbstractObservationOperator[; sensor_noise=zero(ystar)])

Plot the sensor data in `ystar`.

  plot_sensor_data(ystar::Vector,x::Vector,t,obs::AbstractObservationOperator,gridConfig::constructGrids[; sensor_noise=zero(ystar)])

Compute the sensor data associated with state vector `x` at time `t` and plot it with the sensor data in `ystar`.
"""
function plot_sensor_data(a...; sensor_noise=zero(ystar))
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel="Sensor no.",ylabel="Sensor value")
    plot_sensor_data!(ax,a...;sensor_noise=sensor_noise)
    f
end

function plot_sensors!(ax,obs::AbstractObservationOperator{Nx,Ny,true};kwargs...) where {Nx,Ny}
    scatter!(ax,real.(obs.sens),imag.(obs.sens);marker=:rect,color=:black,kwargs...)
end

function draw_ellipse!(ax,μ::Vector,Σ::AbstractMatrix;fill=false,kwargs...)
    xe, ye = get_ellipse_coords(μ,Σ)
    _draw_ellipse!(ax,xe,ye,Val(fill);kwargs...)
end

function _draw_ellipse!(ax,xe,ye,::Val{false};kwargs...)
    lines!(ax,xe,ye;marker=:none,kwargs...)
end

function _draw_ellipse!(ax,xe,ye,::Val{true};color=:red,kwargs...)
    pts = Point2f[(x,y) for (x,y) in zip(xe,ye)]
    poly!(ax,pts;kwargs...)
end

function draw_ellipse_x!(ax,μ::Vector,Σ::AbstractMatrix,z;kwargs...)
    xe, ye = get_ellipse_coords(μ,Σ)
    lines!(ax,z*ones(length(xe)),xe,ye;marker=:none,kwargs...)
end

function draw_ellipse_y!(ax,μ::Vector,Σ::AbstractMatrix,z;kwargs...)
    xe, ye = get_ellipse_coords(μ,Σ)
    lines!(ax,ye,z*ones(length(xe)),xe;marker=:none,kwargs...)
end

function draw_ellipse_z!(ax,μ::Vector,Σ::AbstractMatrix,z;kwargs...)
    xe, ye = get_ellipse_coords(μ,Σ)
    lines!(ax,xe,ye,z*ones(length(xe));marker=:none,kwargs...)
end

function get_ellipse_coords(μ::Vector,Σ::AbstractMatrix)
    θ = range(0,2π,length=100)
    xc, yc = cos.(θ), sin.(θ)
    sqrtΣ = sqrt(Σ)
    xell = μ[1] .+ sqrtΣ[1,1]*xc .+ sqrtΣ[1,2]*yc
    yell = μ[2] .+ sqrtΣ[2,1]*xc .+ sqrtΣ[2,2]*yc
    return xell, yell
end
