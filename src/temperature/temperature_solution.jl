export setup_prob_and_sys, TemperatureSolution

@ilmproblem DirichletPoisson scalar
@ilmproblem NeumannPoisson scalar

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

mutable struct NeumannPoissonCache{SMT,DVT,VNT,FT,ST,FRT} <: ImmersedLayers.AbstractExtraILMCache
    S :: SMT
    dvn :: DVT
    vn :: VNT
    fstar :: FT
    sstar :: ST
    forcing_cache :: FRT
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

function ImmersedLayers.prob_cache(prob::NeumannPoissonProblem,base_cache::BasicILMCache)
    @unpack phys_params, forcing = prob
    S = create_CLinvCT(base_cache)
    dvn = zeros_surface(base_cache)
    vn = zeros_surface(base_cache)
    fstar = zeros_grid(base_cache)
    sstar = zeros_gridcurl(base_cache)
    forcing_cache = ForcingModelAndRegion(forcing["heating models"],base_cache)
    NeumannPoissonCache(S,dvn,vn,fstar,sstar,forcing_cache)
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
    df = zeros_surface(base_cache)

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

    return f, C^6*s, df
end

function ImmersedLayers.solve(prob::NeumannPoissonProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache, bc, phys_params, forcing = sys
    @unpack gdata_cache = base_cache
    @unpack S, dvn, vn, fstar, sstar, forcing_cache = extra_cache

    fill!(fstar,0.0)
    fill!(sstar,0.0)

    f = zeros_grid(base_cache)
    s = zeros_gridcurl(base_cache)
    df = zeros_surface(base_cache)
    ds = zeros_surface(base_cache)

    # apply_forcing! evaluates the forcing field on the grid and put
    # the result in the `gdata_cache`.
    fill!(gdata_cache,0.0)
    apply_forcing!(gdata_cache,f,0.0,forcing_cache,phys_params)

    # Get the precribed jump and average of the surface normal derivatives
    prescribed_surface_jump!(dvn,sys)
    prescribed_surface_average!(vn,sys)

    # Find the potential
    regularize!(fstar,dvn,base_cache)
    fstar .+= gdata_cache
    fstar = -fstar
    inverse_laplacian!(fstar,base_cache)

    surface_grad!(df,fstar,base_cache)
    df .= vn + df
    df .= (S\df);

    surface_divergence!(f,df,base_cache)
    inverse_laplacian!(f,base_cache)
    f .+= fstar

    # Find the streamfunction
    surface_curl!(sstar,df,base_cache)

    surface_grad_cross!(ds,fstar,base_cache)
    ds .= S\ds

    surface_curl_cross!(s,ds,base_cache)
    s .-= sstar
    s .*= -1.0

    inverse_laplacian!(s,base_cache)

    return f, s, df
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

function get_linear_interpolate(prob::NeumannPoissonCache,sys::ILMSystem,Ts::AbstractVector;sensor_boundary="top")
    @unpack base_cache = sys
    @unpack bl, pts = base_cache
    body = bl[1]
    pts_x = pts.u
    pts_y = pts.v
    points_dict = Dict("bottom"=>pts_x[body.side[1]], "right"=>pts_y[body.side[2]], "top"=>sort(pts_x[body.side[3]]), "left"=>sort(pts_y[body.side[4]]))
    T_dict = Dict("bottom"=>Ts[body.side[1]], "right"=>Ts[body.side[2]], "top"=>sort(Ts[body.side[3]]), "left"=>sort(Ts[body.side[4]]))
    return linear_interpolation(points_dict[sensor_boundary], T_dict[sensor_boundary])
end

function setup_prob_and_sys(x::AbstractVector,gridConfig::constructGrids,body,bcdict::Dict,config::HeaterConfig;bc_type="Dirichlet")
	@unpack g, Nθ = gridConfig
    forcing_dict = forcing_region(x,Nθ,config)
    if bc_type == "Dirichlet"
        prob = DirichletPoissonProblem(g,body,scaling=GridScaling,bc=bcdict,forcing=forcing_dict)
    elseif bc_type == "Neumann"
        prob = NeumannPoissonProblem(g,body,scaling=GridScaling,bc=bcdict,forcing=forcing_dict)
    end
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
    f, s, df = solve(prob,sys)
    return f
end

"""
TemperatureSolution(x::AbstractVector,gridConfig::constructGrids,obs::TemperatureObservations,prob,sys::ILMSystem)-> Vector{float64}
solves the temperature poisson equation numerically on grid g, and returns the value of temperature at the
location of sensors imposed by Nq number of heaters.
"""
function TemperatureSolution(x::AbstractVector,gridConfig::constructGrids,obs::TemperatureObservations,prob::DirichletPoissonProblem,sys::ILMSystem)
	@unpack config, sens = obs
	@unpack g, cache, Nθ = gridConfig
	Ny = length(sens)
	T_sens = zeros(Ny)

    forcing_dict = forcing_region(x,Nθ,config)
	sys.extra_cache.forcing_cache = ForcingModelAndRegion(forcing_dict["heating models"],sys.base_cache)
    f, s, df = solve(prob,sys)
    Tfield = interpolatable_field(f,g)
	T_sens .= [Tfield(real(sens[j]), imag(sens[j])) for j in 1:Ny]
    return T_sens
end

function TemperatureSolution(x::AbstractVector,gridConfig::constructGrids,obs::TemperatureObservations,prob::NeumannPoissonProblem,sys::ILMSystem)
	@unpack config, sens = obs
	@unpack g, cache, Nθ = gridConfig
    @unpack base_cache = sys
    @unpack pts = base_cache

	Ny = length(sens)
	T_sens = zeros(Ny)
    Ts = zeros_surface(base_cache)

    forcing_dict = forcing_region(x,Nθ,config)
	sys.extra_cache.forcing_cache = ForcingModelAndRegion(forcing_dict["heating models"],sys.base_cache)
    f, s, df = solve(prob,sys)
    ImmersedLayers.interpolate!(Ts,f,base_cache)
    Ts .+= df./2
	#x_node = collect(pts.u)
    #y_node = collect(pts.v)
    #spl = Spline2D(x_node, y_node, collect(Ts); kx=5, ky=5, s=0.2)
    Tfield = get_linear_interpolate(prob,sys,Ts)
    T_sens .= [Tfield(real(sens[j])) for j in 1:Ny]
	#T_sens .= [evaluate(spl, real(sens[j]), imag(sens[j])) for j in 1:Ny]
    return T_sens
end

