export constructGrids

struct constructGrids{XT,CT} <: AbstractGrids

    "Grid spacing"
    Δx::Float64

    "size of domain in x-direction"
    xlim::XT

    "size of domain in y-direction"
    ylim::XT
    
    "Physical Grid"
    g::PhysicalGrid

    "Cache"
    cache::CT

    "Number of points on heater boundary"
    Nθ::Int64
end

function constructGrids(Δx,bounds;Nθ=500)
    xlim = bounds[1]
    ylim = bounds[2]
    g = PhysicalGrid(xlim,ylim,Δx)
    cache = SurfaceScalarCache(g)
    CacheType = typeof(cache)
    constructGrids{typeof(xlim), CacheType}(Δx,xlim,ylim,g,cache,Nθ)
end