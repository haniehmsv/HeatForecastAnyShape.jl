module HeatForecastAnyShape

using ImmersedLayers
using Interpolations
using JLD
using LinearAlgebra
using ProgressMeter
using Statistics
using NamedColors
using Distributions
using Combinatorics
using UnPack
using Dierckx

abstract type AbstractConfig end
abstract type AbstractGrids end
abstract type SingularityConfig <: AbstractConfig end



include("ensemble.jl")
include("forecast.jl")
include("observation.jl")

include("DA/types.jl")
include("DA/generate_twin_experiment.jl")
include("DA/enkf.jl")
include("DA/state_utilities.jl")
include("DA/classification.jl")
include("DA/MCMC.jl")

include("temperature/grids.jl")
include("heater/heater.jl")
include("heater/heater_clusters.jl")
include("heater/heater_forecast_observation.jl")

include("temperature/jacobian.jl")
include("temperature/temperature_solution.jl")

include("experiments/routines.jl")



include("plot_recipes.jl")






end # module
