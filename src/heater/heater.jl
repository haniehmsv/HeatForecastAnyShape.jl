export HeaterConfig, state_length, construct_heater_state_mapping, number_of_singularities,
          state_to_positions_and_strengths, positions_and_strengths_to_state, get_config_and_state,
          state_covariance, create_state_bounds, get_singularity_ids, state_to_singularity_states,
          states_to_singularity_states, heaterfield



"""
    HeaterConfig

A structure to hold the parameters of the heater simulations

## Fields
- `Nq::Int64`: number of heaters
- `state_id`: Look-up for state component indices
- `Δt::Float64`: time step
"""
struct HeaterConfig{SID} <: SingularityConfig

    "Number of heaters"
    Nq::Int64

    "State IDs"
    state_id::SID

    "Time step"
    Δt::Float64

end

function HeaterConfig(Nq,Δt)
  state_id = construct_heater_state_mapping(Nq)
  HeaterConfig{typeof(state_id)}(Nq,state_id,Δt)
end

function HeaterConfig(Nq)
  state_id = construct_heater_state_mapping(Nq)
  HeaterConfig{typeof(state_id)}(Nq,state_id,0.0)
end



# Mapping from heater elements and other degrees of freedom to the state components
#construct_heater_state_mapping(Nq) = construct_heater_state_mapping(Nq)
function construct_heater_state_mapping(Nq::Int64)
  state_id = Dict()
  heater_x_ids = zeros(Int,Nq)
  heater_y_ids = zeros(Int,Nq)
  heater_q_ids = zeros(Int,Nq)
  heater_c1_ids = zeros(Int,Nq)
  heater_c2_ids = zeros(Int,Nq)

  for j in 1:Nq
    heater_x_ids[j] = 5j-4
    heater_y_ids[j] = 5j-3
    heater_q_ids[j] = 5j-2
    heater_c1_ids[j] = 5j-1
    heater_c2_ids[j] = 5j
  end

  state_id["heater x"] = heater_x_ids
  state_id["heater y"] = heater_y_ids
  state_id["heater q"] = heater_q_ids
  state_id["heater c1"] = heater_c1_ids
  state_id["heater c2"] = heater_c2_ids

  return state_id
end


"""
    state_length(config::AbstractConfig)

Return the number of components of the state vector
"""
state_length(config::AbstractConfig) =  state_length(config.state_id)

state_length(a::Dict) = mapreduce(key -> state_length(a[key]),+,keys(a))
state_length(a::Vector) = length(a)
state_length(a::Matrix) = 0
state_length(a::Int) = 0


number_of_singularities(config::HeaterConfig) = config.Nq

function get_config_and_state(zq::AbstractVector,qq::AbstractVector,c1q::AbstractVector,c2q::AbstractVector)
  Nq = length(zq)
  config = HeaterConfig(Nq)
  x = positions_and_strengths_to_state(zq,qq,c1q,c2q,config)

  return config, x
end


"""
    state_to_positions_and_strengths(state::AbstractVector,config::HeaterConfig) -> Vector{ComplexF64}, Vector{Float64}, Vector{Float64}

Convert the state vector `state` to vectors of positions and strengths.
"""

function state_to_positions_and_strengths(state::AbstractVector{Float64}, config::HeaterConfig)
  @unpack Nq, state_id = config

  x_ids = state_id["heater x"]
  y_ids = state_id["heater y"]
  q_ids = state_id["heater q"]
  c1_ids = state_id["heater c1"]
  c2_ids = state_id["heater c2"]

  zq = [state[x_ids[i]] + im*state[y_ids[i]] for i in 1:Nq]
  qq = [state[q_ids[i]] for i in 1:Nq]
  c1q = [state[c1_ids[i]] for i in 1:Nq]
  c2q = [state[c2_ids[i]] for i in 1:Nq]

  return zq, qq, c1q, c2q
end

function positions_and_strengths_to_state(zq::AbstractVector{ComplexF64},qq::AbstractVector{Float64},c1q::AbstractVector{Float64},c2q::AbstractVector{Float64},config::HeaterConfig)
  @unpack Nq, state_id = config
  state = zeros(state_length(state_id))

  x_ids = state_id["heater x"]
  y_ids = state_id["heater y"]
  q_ids = state_id["heater q"]
  c1_ids = state_id["heater c1"]
  c2_ids = state_id["heater c2"]


  for i = 1:Nq
    state[x_ids[i]] = real(zq[i])
    state[y_ids[i]] = imag(zq[i])
    state[q_ids[i]] = qq[i]
    state[c1_ids[i]] = c1q[i]
    state[c2_ids[i]] = c2q[i]
  end

  return state
end


"""
    state_covariance(varx, vary, varq, varr, config::HeaterConfig)

Create a state covariance matrix with variances `varx`, `vary`, `varq`, and 'varr'
for the x, y, strength, and radius entries for every heater.
"""
function state_covariance(varx, vary, varq, varc, config::HeaterConfig)
  @unpack Nq, state_id = config

  x_ids = state_id["heater x"]
  y_ids = state_id["heater y"]
  q_ids = state_id["heater q"]
  c1_ids = state_id["heater c1"]
  c2_ids = state_id["heater c2"]

  Σx_diag = zeros(Float64,state_length(config))
  for j = 1:Nq
    Σx_diag[x_ids[j]] = varx
    Σx_diag[y_ids[j]] = vary
    Σx_diag[q_ids[j]] = varq
    Σx_diag[c1_ids[j]] = varc
    Σx_diag[c2_ids[j]] = varc
  end

  return Diagonal(Σx_diag)
end

"""
    create_state_bounds(xr::Tuple,yr::Tuple,strengthr::Tuple,rr::Tuple,config::SingularityConfig)

Create a vector of tuples (of length equal to the state vector) containing the
bounds of each type of vector component.
"""
function create_state_bounds(xr,yr,strengthr,c1r,c2r,config::HeaterConfig)
    @unpack state_id = config
    N = number_of_singularities(config)

    bounds = [(-Inf,Inf) for i in 1:state_length(config)]

    x_ids = state_id["heater x"]
    y_ids = state_id["heater y"]
    q_ids = state_id["heater q"]
    c1_ids = state_id["heater c1"]
    c2_ids = state_id["heater c2"]


    for j = 1:N
      bounds[x_ids[j]] = xr
      bounds[y_ids[j]] = yr
      bounds[q_ids[j]] = strengthr
      bounds[c1_ids[j]] = c1r
      bounds[c2_ids[j]] = c2r
    end

    return bounds
end


### OTHER WAYS OF DECOMPOSING STATES ###

"""
    states_to_singularity_states(state_array::Matrix,config::AbstractConfig)

Take an array of states (length(state) x nstates) and convert it to a (5 x Nq*nstates) array
of individual heater states.
"""
function states_to_singularity_states(state_array::AbstractMatrix{Float64}, config::SingularityConfig)
   Ns = number_of_singularities(config)
   ndim, nstates = size(state_array)
   sing_array = zeros(5,Ns*nstates)
   for j in 1:nstates
     singstatej = state_to_singularity_states(state_array[:,j],config)
     sing_array[:,(j-1)*Ns+1:j*Ns] = singstatej
   end
   return sing_array
end

"""
    index_of_singularity_state(h::Integer,config::HeaterConfig)

Return the column of a (length(state) x nstates) array of states
that a single singularity of index `h` in a (5 x Nq*nstates) singularity state array belongs to.
"""
index_of_singularity_state(h::Int,config::AbstractConfig) = (h-1)÷number_of_singularities(config)+1

"""
    state_to_singularity_states(state::AbstractVector,config::HeaterConfig)

Given a state `state`, return a 5 x Nq array of the singularity states.
"""
function state_to_singularity_states(state::AbstractVector{Float64}, config::SingularityConfig)
    Nq = number_of_singularities(config)
    zq, qq, c1q, c2q = state_to_positions_and_strengths(state,config)
    xarray = zeros(5,Nq)
    for h in 1:Nq
        zq_q = zq[h]
        xarray[:,h] .= [real(zq_q),imag(zq_q),qq[h],c1q[h],c2q[h]]
    end
    return xarray
end

state_to_singularity_states(state_array::BasicEnsembleMatrix, config::HeaterConfig) =
    state_to_singularity_states(state_array.X,config)

"""
    get_singularity_ids(config) -> Tuple{Int}

Return the global position and strength IDs in the state vector as
a tuple of 5 integers (e.g. xid, yid, qid, c1id, c2id)
"""
function get_singularity_ids(config::HeaterConfig)
  @unpack state_id = config

  x_ids = state_id["heater x"]
  y_ids = state_id["heater y"]
  q_ids = state_id["heater q"]
  c1_ids = state_id["heater c1"]
  c2_ids = state_id["heater c2"]

  return x_ids, y_ids, q_ids, c1_ids, c2_ids
end

"""
    get_singularity_ids(h,config) -> Tuple{Int}

Return the global position and strength IDs in the state vector as
a tuple of 5 integers (e.g. xid, yid, qid, c1id, c2id)
"""
function get_singularity_ids(h::Integer,config::HeaterConfig)
  @unpack state_id = config
  Nq = number_of_singularities(config)

  @assert h <= Nq && h > 0

  x_ids = state_id["heater x"]
  y_ids = state_id["heater y"]
  q_ids = state_id["heater q"]
  c1_ids = state_id["heater c1"]
  c2_ids = state_id["heater c2"]

  return x_ids[h], y_ids[h], q_ids[h], c1_ids[h], c2_ids[h]
end

"""
    heaterfield(x,y,μ::AbstactVector,Σ::AbstractMatrix,config::HeaterConfig)

Evaluate the heaterfield at point x,y, given the mean state vector `μ` and uncertainty matrix `Σ`
"""
function heaterfield(x,y,μ::Vector,Σ::AbstractMatrix,config::SingularityConfig)
    @unpack state_id = config
    Nq = number_of_singularities(config)

    x_ids, y_ids, q_ids, c1_ids, c2_ids = get_singularity_ids(config)

    xvec = [x,y]
    heat = 0.0
    for j = 1:Nq
        xidj, yidj, qidj, c1idj, c2idj = get_singularity_ids(j,config)
        μxj = μ[[xidj,yidj]]
        Σxxj = Σ[xidj:yidj,xidj:yidj]
        Σqxj = Σ[xidj:yidj,qidj]
        Σc1xj = Σ[xidj:yidj,c1idj]
        Σc2xj = Σ[xidj:yidj,c2idj]
        μqj = μ[qidj]
        μc1j = μ[c1idj]
        μc2j = μ[c2idj]
        heatj = _heaterfield(xvec,μxj,μqj,Σxxj,Σqxj)
        heat += heatj
    end
    return heat
end


function _heaterfield(xvec, μx, μq, Σxx, Σqx)
    xvec_rel = xvec .- μx
    heat = exp(-0.5*xvec_rel'*inv(Σxx)*xvec_rel)
    heat *= (μq + Σqx'*inv(Σxx)*xvec_rel)
    return heat
end
