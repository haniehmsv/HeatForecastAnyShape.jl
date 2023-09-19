### Forecasting, observation, filtering, and localization operators for vortex problems ####

export HeaterForecast, TemperatureObservations, physical_space_sensors

#import TransportBasedInference: Parallel, Serial, Thread # These should not be necessary


#### FORECAST OPERATORS ####


struct HeaterForecast{Nx} <: AbstractForecastOperator{Nx}
		config :: HeaterConfig
end

"""
		HeaterForecast(config::HeaterConfig)

Allocate the structure for forecasting of heaters
"""
function HeaterForecast(config::HeaterConfig)
	Nx = 5*config.Nq
	HeaterForecast{Nx}(config)
end

struct SymmetricHeaterForecast{Nx} <: AbstractForecastOperator{Nx}
		config :: HeaterConfig
end

"""
		SymmetricHeaterForecast(config::HeaterConfig)

Allocate the structure for forecasting of heaters with symmetry
about the x axis.
"""
function SymmetricHeaterForecast(config::HeaterConfig)
	Nx = 5*config.Nq
	SymmetricHeaterForecast{Nx}(config)
end

#### OBSERVATION OPERATORS ####

abstract type AbstractCartesianHeaterObservations{Nx,Ny} <: AbstractObservationOperator{Nx,Ny,true} end


#### BEGIN DEFINING TEMPERATURE OBSERVATIONS #####


# Temperature

struct TemperatureObservations{Nx,Ny,ST,CT} <: AbstractObservationOperator{Nx,Ny,true}
	sens::ST
	config::CT
end

"""
	TemperatureObservations(sens::AbstractVector,config::HeaterConfig)

Constructor to create an instance of temperature sensors. The locations of the
sensors are specified by `sens`, which should be given as a vector of
complex coordinates.
"""
function TemperatureObservations(sens::AbstractVector,config::HeaterConfig)
    return TemperatureObservations{5*config.Nq,length(sens),typeof(sens),typeof(config)}(sens,config)
end

function observations(x::AbstractVector,t,obs::TemperatureObservations,gridConfig::constructGrids,prob,sys::ILMSystem)
  return _temperature(x,obs,gridConfig,prob,sys)
end

function jacob!(J,x::AbstractVector,t,obs::TemperatureObservations)
    @unpack config, sens = obs
    return _temperature_jacobian!(J,sens,x,config)
end

_temperature(x,obs::TemperatureObservations,gridConfig::constructGrids,prob,sys::ILMSystem) = TemperatureSolution(x,gridConfig,obs,prob,sys)
_temperature_jacobian!(J,sens,x,config::HeaterConfig) = temperature_jacobian!(J,sens,x,config)


physical_space_sensors(obs::TemperatureObservations) = physical_space_sensors(obs.sens,obs.config)
physical_space_sensors(sens,config::HeaterConfig) = sens



"""
		state_filter!(x,obs::TemperatureObservations)

A filter function to ensure that the heaters stay above the x-axis, and retain a positive strength.
This function would typically be used before and after the analysis step to enforce those constraints.
"""
state_filter!(x, obs::TemperatureObservations) = flip_symmetry_state_filter!(x, obs.config)

function flip_symmetry_state_filter!(x, config::HeaterConfig)
	@unpack Nq, state_id = config

	x_ids = state_id["heater x"]
	y_ids = state_id["heater y"]
	q_ids = state_id["heater q"]
	c1_ids = state_id["heater c1"]
	c2_ids = state_id["heater c2"]

	# Flip the sign of heater if it is negative on average
	qtot = sum(x[q_ids])
	x[q_ids] .= qtot < 0 ? -x[q_ids] : x[q_ids]

	# Sort the heaters by strength to try to ensure they don't take each other's role
	id_sort = sortperm(x[q_ids])
	x[x_ids] .= x[x_ids[id_sort]]
	x[y_ids] .= x[y_ids[id_sort]]
	x[q_ids] .= x[q_ids[id_sort]]
	x[c1_ids] .= x[c1_ids[id_sort]]
	x[c2_ids] .= x[c2_ids[id_sort]]

	# Make all y locations positive
	#x[y_ids] = abs.(x[y_ids])

  return x
end



### LOCALIZATION ###

function dobsobs(obs::AbstractCartesianHeaterObservations{Nx,Ny}) where {Nx,Ny}
		@unpack sens, config = obs
    dYY = zeros(Ny, Ny)
    # Exploit symmetry of the distance matrix dYY
    for i=1:Ny
        for j=1:i-1
            dij = abs(sens[i] - sens[j])
            dYY[i,j] = dij
            dYY[j,i] = dij
        end
    end
    return dYY
end

function dstateobs(X::BasicEnsembleMatrix{Nx,Ne}, obs::AbstractCartesianHeaterObservations{Nx,Ny}) where {Nx,Ny,Ne}
#function dstateobs(X, obs::AbstractCartesianHeaterObservations{Nx,Ny}) where {Nx,Ny}

	@unpack config, sens = obs
	@unpack Nq, state_id = config

	x_ids = state_id["heater x"]
	y_ids = state_id["heater y"]
	q_ids = state_id["heater q"]
	c1_ids = state_id["heater c1"]
	c2_ids = state_id["heater c2"]

	dXY = zeros(Nx, Ny)
	for i in 1:Ne
		xi = X(i)
		zi = map(l->xi[x_ids[l]] + im*xi[y_ids[l]], 1:Nq)

		for J in 1:Nq
				for k in 1:Ny
						dXY[J,k] += abs(zi[J] - sens[k])
				end
		end
	end
	dXY ./= Ne
	return dXY
end

function apply_state_localization!(Σxy,X,Lxy,obs::AbstractCartesianHeaterObservations)
	@unpack config = obs
	@unpack Nq, state_id = config
  dxy = dstateobs(X, obs)
  Gxy = gaspari.(dxy./Lxy)

	x_ids = state_id["heater x"]
	y_ids = state_id["heater y"]
	q_ids = state_id["heater q"]
	c1_ids = state_id["heater c1"]
	c2_ids = state_id["heater c2"]

  for J=1:Nq
		Σxy[x_ids[J],:] .*= Gxy[J,:]
		Σxy[y_ids[J],:] .*= Gxy[J,:]
		Σxy[q_ids[J],:] .*= Gxy[J,:]
		Σxy[c1_ids[J],:] .*= Gxy[J,:]
		Σxy[c2_ids[J],:] .*= Gxy[J,:]
  end
  return nothing
end
