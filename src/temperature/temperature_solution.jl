export analytical_temperature

"""
analytical_temperature(x::AbstractVector,obs::TemperatureObservations,g::PhysicalGrid)-> Vector{float64}
solves the temperature poisson equation numerically on grid g, and returns the value of temperature at the
location of sensors imposed by Nq number of heaters.
"""

function analytical_temperature(x::AbstractVector,obs::TemperatureObservations,gridConfig::constructGrids)
	@unpack config, sens = obs
	@unpack Nq, state_id = config
	@unpack g, cache, Ntheta = gridConfig
	Ny = length(sens)

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

	T = zeros_grid(cache)
	xg, yg = coordinates(T,g)
	Temp = zeros(Ny)

	@inbounds for k in 1:Nq
		r_real, r_imag = create_points_on_shape(x[5k-4:5k],gridConfig)
		T = zeros_grid(cache)
		theta_g = zeros_grid(cache)
		@inbounds for (j,yy) in enumerate(yg), (i,xx) in enumerate(xg)
			theta_g[i,j] = (yy-yq[k])>=0 ? atan((yy-yq[k]),(xx-xq[k])) : (atan((yy-yq[k]),(xx-xq[k]))+2π)
			if ((xx-xq[k])^2+(yy-yq[k])^2) < ((r_real(theta_g[i,j])-xq[k])^2 + (r_imag(theta_g[i,j])-yq[k])^2)
				T[i,j] = -qq[k]
			end
		end
		inverse_laplacian!(T,cache)
		Tfield = interpolatable_field(T,g)
		Temp .+= [Tfield(real(sens[j]), imag(sens[j])) for j in 1:Ny]
	end
	return Temp
end


"""
analytical_temperature(x::AbstractVector,obs::TemperatureObservations,g::PhysicalGrid,matrix_output::Bool)-> Matrix{float64}
solves the temperature poisson equation numerically on grid g, and returns the value of temperature at all
grid points in the domain.
"""

function analytical_temperature(x::AbstractVector,obs::TemperatureObservations,gridConfig::constructGrids,matrix_output::Bool)
	@unpack config = obs
	@unpack Nq, state_id = config
	@unpack g, cache, Ntheta = gridConfig

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

	T = zeros_grid(cache)
	Temp = zeros_grid(cache)
	xg, yg = coordinates(T,g)

	@inbounds for k in 1:Nq
		r_real, r_imag = create_points_on_shape(x[5k-4:5k],gridConfig)
		T = zeros_grid(cache)
		theta_g = zeros_grid(cache)
		@inbounds for (j,yy) in enumerate(yg), (i,xx) in enumerate(xg)
			theta_g[i,j] = (yy-yq[k])>=0 ? atan((yy-yq[k]),(xx-xq[k])) : (atan((yy-yq[k]),(xx-xq[k]))+2π)
			if ((xx-xq[k])^2+(yy-yq[k])^2) < ((r_real(theta_g[i,j])-xq[k])^2 + (r_imag(theta_g[i,j])-yq[k])^2)
				T[i,j] = -qq[k]
			end
		end
		inverse_laplacian!(T,cache)
		Temp .+= T
	end
	return Temp
end