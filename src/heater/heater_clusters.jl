export create_random_heaters, generate_random_state, generate_random_states,
          createclusters, heatercluster

#=
function heatermoment(mom::Int,zq::AbstractVector,strengthq::AbstractVector,c1q::AbstractVector,c2q::AbstractVector)
    @assert length(zq) == length(strengthq) == length(c1q) == length(c2q) "Inconsistent lengths of vectors"
    sum = complex(0.0)
    for j in 1:length(zq)
      sum += strengthq[j]*zq[j]^mom*radiusq[j]^(2*mom)
    end
    return sum
end
=#

# Create a set of n random heaters in the range [-4,4]x[-4,4]
function create_random_heaters(n::Integer,xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple)
    z = ComplexF64[]
    q = Float64[]
    c1 = Float64[]
    c2 = Float64[]
    num = 0
    while num < n
        ztest = (xr[1] + rand(Float64)*(xr[2]-xr[1])) + im*(yr[1] + rand(Float64)*(yr[2]-yr[1]))
        qtest = qr[1] + rand(Float64)*(qr[2]-qr[1])
        c1test = c1r[1] + rand(Float64)*(c1r[2]-c1r[1])
        c2test = c1r[1] + rand(Float64)*(c2r[2]-c2r[1])
        if (abs2(ztest)>=1.0)
            push!(z,ztest)
            push!(q,qtest)
            push!(c1,c1test)
            push!(c2,c2test)
            num += 1
        end
    end
    return z,q,c1,c2
end

"""
    generate_random_state(xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple,config::HeaterConfig) -> Vector{Float64}

Generate a random state, taking the parameters for the heaters from the ranges
`xr`, `yr`, `qr`, 'c1r', 'c2r'.
"""
function generate_random_state(xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple,config::SingularityConfig)
    Nq = number_of_singularities(config)
    zq_prior = random_points_plane(Nq,xr...,yr...)
    strengthq_prior = random_strengths(Nq,qr...)
    c1q_prior = random_coefficient1(Nq,c1r...)
    c2q_prior = random_coefficient2(Nq,c1q_prior,c2r...)
    return positions_and_strengths_to_state(zq_prior,strengthq_prior,c1q_prior,c2q_prior,config)
end

generate_random_states(nstates,a...) = [generate_random_state(a...) for i in 1:nstates]

generate_random_state(bounds::Vector,config::SingularityConfig) = generate_random_state(bounds[1:5]...,config)


"""
    createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple,σx,σq,σc;each_cluster_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}, Vector{Float64}, Vector{Float64}

Return `Nclusters` clusters of heaters, their strengths, and Fourier coefficients. The centers of the clusters are distributed
randomly in the box `xr` x `yr`. Their total strengths
are uniformly distributed in the range `qr`. Their coefficients
are uniformly distributed in the range `cr`. Within each cluster,
the heaters are distributed randomly about the cluster center accordingly to `σx`, with strengths
randomly distributed about the cluster strength (divided equally), with standard deviation `σq`, and coefficients
of heaters are distributed randomly about the heater coeffs with standard deviation 'σc'.
"""
createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple,σx,σq,σc;each_cluster_radius=0.0) =
      _createclusters(Nclusters,Npercluster,xr,yr,qr,c1r,c2r,σx,σq,σc,each_cluster_radius)


function _createclusters(Nclusters,Npercluster,xr::Tuple,yr::Tuple,qr::Tuple,c1r::Tuple,c2r::Tuple,σx,σq,σc,each_cluster_radius)
    zc = random_points_plane(Nclusters,xr...,yr...)
    qc = random_strengths(Nclusters,qr...)
    c1c = random_coefficient1(Nclusters,c1r...)
    c2c = random_coefficient1(Nclusters,c2r...)

    zp = ComplexF64[]
    qp = Float64[]
    c1p = Float64[]
    c2p = Float64[]
    for j in 1:Nclusters
        zpj, qpj, c1pj, c2pj = heatercluster(Npercluster,zc[j],qc[j],c1c[j],c2c[j],σx,σq,σc;circle_radius=each_cluster_radius)
        append!(zp,zpj)
        append!(qp,qpj)
        append!(c1p,c1pj)
        append!(c2p,c2pj)
    end
    return zp, qp, c1p, c2p
end



"""
    heatercluster(N,z0,q0,c10,c20,σx,σq,σc;circle_radius=0.0) -> Vector{ComplexF64}, Vector{Float64}, Vector{Float64}, Vector{Float64}

Create a cluster of `N` heaters, distributed randomly about the center `z0` according to standard
deviation `σx`, each with strengths deviating randomly by `σq` from `q0` divided equally about the heaters,
and each with sizes deviating randomly by`σc` from 'c10' and 'c20'.
"""
function heatercluster(N,z0,q0,c10,c20,σx,σq,σc;circle_radius=0.0)
    @assert circle_radius >= 0.0 "circle_radius must be non-negative"
    dx = Normal(0.0,σx)
    dq = Normal(0.0,σq)
    dc = Normal(0.0,σc)

    θv = range(0,2π,length=N+1)[1:end-1] .+ π/2*mod(N,2)
    zcirc = N > 1 ? circle_radius*exp.(im*θv) : zeros(ComplexF64,1)
    zp = z0 .+ zcirc
    
    zp .+= rand(dx,N) .+ im*rand(dx,N)
    qp = q0/N*ones(N)
    qp .+= rand(dq,N)
    c1p = c10
    c2p = c20
    c1p .+= rand(dc,N)
    c2p .+= rand(dc,N)

    return zp, qp, c1p, c2p
end

"""
    random_points_plane(N,xmin,xmax,ymin,ymax) -> Vector{ComplexF64}

Create a set of `N` random points in the plane, uniformly
distributed in the complex plane inside the region [xmin,xmax] x [ymin,ymax]
"""
function random_points_plane(N,xmin,xmax,ymin,ymax)
    @assert xmax > xmin "xmax must be larger than xmin"
    @assert ymax > ymin "ymax must be larger than ymin"
    x = xmin .+ (xmax-xmin)*rand(N)
    y = ymin .+ (ymax-ymin)*rand(N)
    return x .+ im*y
end

function random_strengths(N,qmin,qmax)
  @assert qmax > qmin "qmax must be larger than qmin"
  q = qmin .+ (qmax-qmin)*rand(N)
  return q
end

function random_coefficient1(N,c1min,c1max)
    @assert c1max > c1min "rmax must be larger than rmin"
    c1 = c1min .+ (c1max-c1min)*rand(N)
    return c1
end

function random_coefficient2(N,c1,c2min,c2max)
    @assert c2max > c2min "rmax must be larger than rmin"
    c2 = c2min .+ (c2max-c2min)*rand(N)
    c2 .= min.(0.5 .* c1, c2)
    return c2
end