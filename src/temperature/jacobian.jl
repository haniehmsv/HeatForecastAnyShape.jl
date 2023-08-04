export analytical_temperature_jacobian!


analytical_temperature_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::AbstractVector,config::HeaterConfig;kwargs...) =
			analytical_temperature_jacobian!(J,target,source,config.state_id,config.Nq;kwargs...)

function analytical_temperature_jacobian!(J,target::AbstractVector{<:ComplexF64}, source::AbstractVector, state_id::Dict,Nq::Int64; kwargs...)
	Nx = state_length(state_id)
	Ny = size(target, 1)
  #J = zeros(Float64,Ny,Nx)
  @assert size(J) == (Ny, Nx)

	x_ids = state_id["heater x"]
  	y_ids = state_id["heater y"]
  	q_ids = state_id["heater q"]
	c1_ids = state_id["heater c1"]
	c2_ids = state_id["heater c2"]


	dTdzi = zeros(ComplexF64, Ny)
	dTdqi = zeros(Ny)
	dTdc1i = zeros(Ny)
	dTdc2i = zeros(Ny)

	for i=1:Nq
		J[:,x_ids[i]] .= real.(dTdzi)
		J[:,y_ids[i]] .= imag.(dTdzi)
		J[:,q_ids[i]] .= dTdqi
		J[:,c1_ids[i]] .= dTdc1i
		J[:,c2_ids[i]] .= dTdc2i
	end

	return J
end