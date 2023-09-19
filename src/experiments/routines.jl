export get_truth_data, setup_sensors, setup_estimator

"""
    get_truth_data(Nsens,ϵmeas,x_true,config_true[;layout=(:line,1.0),add_noise=false]) -> obs_true, ystar, H, Σϵ, Σx

Given the true state (for twin experiment) and configuration structure, return
the observations at a given number of sensors `Nsens`, arranged in layout `layout`,
with noise `ϵmeas`. The default layout is a `:line` extending from -1.0 to 1.0 on the x axis.
It can also be a circle, e.g. of radius 2, `layout=(:circle,2.0)`.
By default the noise is not added to the data, but can be if `add_noise` is set to true.
"""
function get_truth_data(Nsens,ϵmeas,x_true::Vector,config_true::SingularityConfig,gridConfig::constructGrids,prob,sys;layout=(:line,1.0),add_noise=false)

    t = 0.0

    sens = setup_sensors(Nsens;layout=layout)

    # Get true observations
    obs_true = TemperatureObservations(sens,config_true)

    # evaluate data without noise
    ystar0 = observations(x_true,t,obs_true,gridConfig,prob,sys)

    Σϵ = Diagonal(ϵmeas^2*ones(length(sens)))

    noisedist = MvNormal(zero(ystar0),Σϵ)
    ystar = ystar0
    if add_noise
        ystar += rand(noisedist)
    end

    n = state_length(config_true)
    H = zeros(Nsens,n)
    jacob!(H,x_true,t,obs_true)

    invΣx = H'*inv(Σϵ)*H
    Σx = rank(invΣx) == n ? inv(invΣx) : Matrix{Float64}(undef,n,n)

    return obs_true, ystar, H, Σϵ, Σx

end

function setup_sensors(Nsens;layout=(:line,1.0))

  layout_type, len = layout

  if layout_type == :circle
    rsens = len
    θsens = range(0,2π,length=Nsens+1)
    sens = rsens*exp.(im*θsens[1:end-1])
  elseif layout_type == :line
    ϵsens = 0.0
    lowerrow = range(-len,len,length=Nsens) .+ (-0.5ϵsens .+ ϵsens*rand(Nsens))*im
    #upperrow = range(-2.0,2.0,length=Nsens) .+ 1.0*im
    #leftside = im*range(-1.0,3.0,length=Nsens) .- 1.0
    #rightside = im*range(-1.0,3.0,length=Nsens) .+ 1.0
    sens = vcat(lowerrow,)  #upperrow);
  elseif layout_type == :dline
    ϵsens = 0.02
    lowerrow1 = range(-len,len,length=Nsens÷2) .- 0.5*ϵsens
    lowerrow2 = range(-len,len,length=Nsens÷2) .+ 0.5*ϵsens
    sens = sort(vcat(lowerrow1,lowerrow2)) .+ 0.0im
  end
  return sens
end

function setup_estimator(sens,bounds,config::SingularityConfig,ystar,Σϵ,gridConfig,prob,sys::ILMSystem)
    t = 0.0

    obs = TemperatureObservations(sens,config)

    function logp̃_fcn(x::Vector)
        logp̃ = normal_loglikelihood(x,t,ystar,Σϵ,obs,gridConfig,prob,sys)
        logp̃ += log_uniform(x,bounds)
    end

    return logp̃_fcn, obs
end
