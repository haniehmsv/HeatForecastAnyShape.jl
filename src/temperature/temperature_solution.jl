export setup_prob_and_sys, TemperatureSolution

@ilmproblem DirichletPoisson scalar

"""
Constructing extra cache for the problem
"""
mutable struct DirichletPoissonCache{SMT,CMT,FRT,ST,FT} <: ImmersedLayers.AbstractExtraILMCache
    S :: SMT
    C :: CMT
    forcing_cache :: FRT
    fb :: ST
    fstar :: FT
 end

"""
  Extending prob_cache function in the ImmersedLayers Pkg to include the extra cache in the problem, for dispatch purposes
"""
function ImmersedLayers.prob_cache(prob::DirichletPoissonProblem,base_cache::BasicILMCache)
    @unpack phys_params, forcing = prob
    S = create_RTLinvR(base_cache)
    C = create_surface_filter(base_cache)
    forcing_cache = ForcingModelAndRegion(forcing["heating models"],base_cache)
    fb = zeros_surface(base_cache)
    fstar = zeros_grid(base_cache)
    DirichletPoissonCache(S,C,forcing_cache,fb,fstar)
end

"""
Extending solve function in the ImmersedLayers Pkg to include the forcing model in the problem, for dispatch purposes
"""
function ImmersedLayers.solve(prob::DirichletPoissonProblem,sys::ILMSystem)
    @unpack bc, forcing, phys_params, extra_cache, base_cache = sys
    @unpack gdata_cache = base_cache
    @unpack S, C, forcing_cache, fb, fstar = extra_cache

    f = zeros_grid(base_cache)
    s = zeros_surface(base_cache)

    # apply_forcing! evaluates the forcing field on the grid and put
    # the result in the `gdata_cache`.
    fill!(gdata_cache,0.0)
    apply_forcing!(gdata_cache,f,0.0,forcing_cache,phys_params)

    # Get the prescribed jump in boundary data across the interface using
    # the functions we supplied via the `Dict`.
    prescribed_surface_jump!(fb,sys)

    # Evaluate the double-layer term and add it to the right-hand side
    surface_divergence!(fstar,fb,base_cache)
    fstar .+= gdata_cache

    # Intermediate solution
    inverse_laplacian!(fstar,base_cache)

    # Get the prescribed average of boundary data on the interface using
    # the functions we supplied via the `Dict`.
    prescribed_surface_average!(fb,sys)

    # Correction
    interpolate!(s,fstar,base_cache)
    s .= fb - s
    s .= -(S\s);

    regularize!(f,s,base_cache)
    inverse_laplacian!(f,base_cache)
    f .+= fstar;

    return f, C^6*s
end

"""
z coordinates of points on the forcing region
"""
function z(θ,c0,c1,c2)
	return c0 + c1*exp(im*θ) + c2*exp(2*im*θ) 
end

"""
	Constructing forcing cache for the problem
"""
function forcing_region(x::AbstractVector,Nθ::Int64,config::HeaterConfig)
    @unpack Nq, state_id = config
    x_ids = state_id["heater x"]
    y_ids = state_id["heater y"]
    q_ids = state_id["heater q"]
    c1_ids = state_id["heater c1"]
    c2_ids = state_id["heater c2"]

    xq = x[x_ids]
    yq = x[y_ids]
    qq = x[q_ids]
    c1q = x[c1_ids]
    c2q = x[c2_ids]

    θ = collect(range(0,2π,length=Nθ))
    pop!(θ)
    afm = [AreaForcingModel(_create_fregion_and_force_model(xq[k],yq[k],qq[k],c1q[k],c2q[k],θ)...) for k in 1:Nq]	#area forcing model
    forcing_dict = Dict{String, Vector{AbstractForcingModel}}("heating models" => afm)
    return forcing_dict
end

function _create_fregion_and_force_model(xqk,yqk,qqk,c1qk,c2qk,θ)
	zp = z.(θ,xqk+im*yqk,c1qk,c2qk)
    x, y = real(zp), imag(zp)
    fregion = BasicBody(x,y)
    function area_strengths!(σ,T,t,fr::AreaRegionCache,phys_params)
            σ .= qqk #phys_params["areaheater_flux"]
    end
	return fregion, area_strengths!
end

function setup_prob_and_sys(x::AbstractVector,gridConfig::constructGrids,body,bcdict::Dict,config::HeaterConfig)
	@unpack g, Nθ = gridConfig
    forcing_dict = forcing_region(x,Nθ,config)
    prob = DirichletPoissonProblem(g,body,scaling=GridScaling,bc=bcdict,forcing=forcing_dict)
    sys = construct_system(prob)
    return prob, sys
end

"""
TemperatureSolution(x::AbstractVector,Nθ::Int64,obs::TemperatureObservations,prob,sys::ILMSystem)-> Matrix{float64}
solves the temperature poisson equation numerically on grid g, and returns the value of temperature at the
grid points.
"""
function TemperatureSolution(x::AbstractVector,Nθ::Int64,obs::TemperatureObservations,prob,sys::ILMSystem)
	@unpack config = obs
    forcing_dict = forcing_region(x,Nθ,config)
	sys.extra_cache.forcing_cache = ForcingModelAndRegion(forcing_dict["heating models"],sys.base_cache)
    f, s = solve(prob,sys)
    return f
end

"""
TemperatureSolution(x::AbstractVector,gridConfig::constructGrids,obs::TemperatureObservations,prob,sys::ILMSystem)-> Vector{float64}
solves the temperature poisson equation numerically on grid g, and returns the value of temperature at the
location of sensors imposed by Nq number of heaters.
"""
function TemperatureSolution(x::AbstractVector,gridConfig::constructGrids,obs::TemperatureObservations,prob,sys::ILMSystem)
	@unpack config, sens = obs
	@unpack g, cache, Nθ = gridConfig
	Ny = length(sens)
	T_sens = zeros(Ny)

    forcing_dict = forcing_region(x,Nθ,config)
	sys.extra_cache.forcing_cache = ForcingModelAndRegion(forcing_dict["heating models"],sys.base_cache)
    f, s = solve(prob,sys)
	Tfield = interpolatable_field(f,g)
	T_sens .= [Tfield(real(sens[j]), imag(sens[j])) for j in 1:Ny]
    return T_sens
end