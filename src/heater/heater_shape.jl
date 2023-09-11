export create_points_on_shape
""" 
create_points_on_shape(x::AbstractVector,Ntheta::Int64) -> Vector{Float64}, Vector{Float64}
A function that generates equaly spaced points on the heater boundary based on Fourier series in complex form: r(θ) = c0 + ∑ (cn exp(inθ))
    where c's are the Fourier coefficients. Here we consider the first three coefficients, c0, c1, and c2. It returns the interpolated points
    to cover all values of θ.
    """
@inline function create_points_on_shape(x::AbstractVector,gridConfig::constructGrids)
    @unpack Ntheta = gridConfig
    theta = collect(range(0,2π,Ntheta))
    pop!(θ)
    c0 = x[1] + im*x[2]
    q = x[3]
    c1 = x[4]
    c2 = x[5]
    r = [c0 + c1*exp(1im*theta[i]) + c2*exp(2im*theta[i]) for i in 1:(Ntheta-1)]
    interpolator_real = LinearInterpolation(theta,real(r))
    interpolator_imag = LinearInterpolation(theta,imag(r))
    return interpolator_real, interpolator_imag
end