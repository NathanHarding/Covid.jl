module simplex_integration

export SimplexIntegration, integrate

using LightGraphs
using LinearAlgebra
using MetaGraphs

const store = Dict{Symbol, Any}()

mutable struct SimplexIntegration
    domain::Vector{Tuple{Float64, Float64}}
    points::Array{Float64, 2}  # ndim x npoints. Each column is a point in Euclidean space
    heights::Vector{Float64}   # heights[i] = f(points[:, j])
    simplices::MetaGraph{Int, Float64}  # Vertex properties: points, integral
    integral::Float64  # The value of the integral

    function SimplexIntegration(domain, points, heights, simplices, integral)
        ndim    = size(domain, 1)
        npoints = size(points, 2)
        size(points, 1)  != ndim    && error("The dimension of each point ($(size(points, 1))) does not match the dimension of the domain ($(ndim)).")
        size(heights, 1) != npoints && error("The number of heights ($(size(heights,1))) does not match the number of points ($(npoints)).")        
        nv(simplices)    != 0       && error("The initial number of vertices should be 0")
        integral         != 0.0     && error("The initial value of the integral should be 0.0")
        new(domain, points, heights, simplices, integral)
    end
end

SimplexIntegration(domain, npoints) = SimplexIntegration(domain, fill(NaN, size(domain, 1), npoints), fill(NaN, npoints), MetaGraph(SimpleDiGraph()), 0.0)

"""
- Integrate siblings before children 
"""
function integrate(f::Function, domain::Vector{Tuple{Float64, Float64}}, npoints::Int, x0::Vector{Float64}, rtol::Float64)
    result    = SimplexIntegration(domain, npoints)
    ndim      = size(domain, 1)
    simplices = result.simplices
    store[:npoints]   = 0  # The number of points measured so far
    store[:notdone]   = Set{Int}()  # Indices of simplicies that may require further splitting
    store[:detmatrix] = fill(0.0, ndim, ndim)  # Used to calculate the volume of a simplex
    index0 = initial_simplex!(result, x0)
    integrate_simplex!(result, f, index0)
    while store[:npoints] < npoints
        index1, index2 = split_simplex!(result, index0)  # Creates a new point and 2 new simplices (nested in the parent)
        integrate_simplex!(result, f, index1)
        integrate_simplex!(result, f, index2)
        parent_integral   = integral(simplices, index0)
        children_integral = integral(simplices, index1) + integral(simplices, index2)
        if parent_integral == 0.0
            relchange = children_integral == 0.0 ? 0.0 : 1.0
        else
            relchange = abs(children_integral / parent_integral - 1.0)
        end
        if relchange <= rtol  # Integral didn't change enough...stop splitting
            pop!(store[:notdone], index1)
            pop!(store[:notdone], index2)
            if isempty(store[:notdone])
                index0 = new_simplex!(result, index0)  # Create a new simplex from simplices[index0]
                integrate_simplex!(result, f, index0)
            else
                index0 = pop!(store[:notdone])  # Split an existing simplex
            end
        else
            index0 = index1  # Split this simplex next
        end
    end
    result.integral = totalintegral!(simplices)
    result
end

"Volume of a simplex."
function volume(simplex::T) where {T <: AbstractArray{Float64, 2}}
    M   = store[:detmatrix]
    n   = size(simplex, 1)
    np1 = n + 1
    for i = 1:n
        for j = 1:n
            M[i, j] = simplex[i, j] - simplex[i, np1]
        end
    end
    abs(det(M)) / factorial(n)
end

"Construct a simplex using x0 and return its index"
function initial_simplex!(result, x0::Vector{Float64})
    points = result.points
    store[:npoints] += length(x0) + 1  # npoints in a simplex is ndim + 1
    ndim = size(points, 1)
    for i = 1:ndim
        points[i, 1] = x0[i]
    end
    alpha = 0.5
    for (j, lb_ub) in enumerate(result.domain)
        lb, ub = lb_ub
        for i = 1:ndim
            if i == j
                points[i, j+1] = lb + alpha * (ub - lb)
            else
                points[i, j+1] = x0[i]
            end
        end
    end
    push_new_simplex!(result.simplices, collect(1:(ndim + 1)))
end

function new_simplex!(result, index0::Int)
    # Compute centroid of simplex that excludes the worst point
    point_indices = get_prop(result.simplices, index0, :points)
    jmin, jmax    = best_worst_vertices(point_indices, result.heights)
    points        = result.points
    ndim          = size(points, 1)
    centroid      = fill(0.0, ndim)
    for j in point_indices
        j == jmin && continue
        for i = 1:ndim
            centroid[i] += points[i, j]
        end
    end
    centroid ./= ndim

    # Creat new point by relecting the worst point about the centroid
    store[:npoints] += 1
    jnew = store[:npoints]
    for i = 1:ndim
        points[i, jnew] = 2.0 * centroid[i] - points[i, jmin]
    end

    # Create new simplex
    newindices = replace(point_indices, jmin => jnew)  # Copy point_indices and replace jmin with jnew
    push_new_simplex!(result.simplices, newindices)
end

"Calculate the height at each vertex and integrate the integrand over the simplex."
function integrate_simplex!(result, f, index)
    point_indices = get_prop(result.simplices, index, :points)
    avgheight = 0.0
    for j in point_indices
        x = view(result.points, :, j)
        h = isnan(result.heights[j]) ? f(x) : result.heights[j]
        avgheight += h
        result.heights[j] = h
    end
    avgheight /= length(point_indices)
    integral = avgheight * volume(view(result.points, :, point_indices))
    set_prop!(result.simplices, index, :integral, integral)
end

function split_simplex!(result, index0)
    # Identify the edge to split
    point_indices = get_prop(result.simplices, index0, :points)
    jmin, jmax    = best_worst_vertices(point_indices, result.heights)

    # Creat a new point by splitting the edge
    store[:npoints] += 1
    jnew   = store[:npoints]
    points = result.points
    ndim   = size(points, 1)
    for i = 1:ndim
        points[i, jnew] = 0.5 * (points[i, jmin] + points[i, jmax])
    end

    # Create 2 child simplices
    simplices = result.simplices
    pop!(store[:notdone], index0)
    newindices = replace(point_indices, jmin => jnew)  # Copy point_indices and replace jmin with jnew
    index1 = push_new_simplex!(simplices, newindices)
    add_edge!(simplices, index0, index1)  # index0 is a parent of index1
    newindices = replace(point_indices, jmax => jnew)  # Copy point_indices and replace jmax with jnew
    index2 = push_new_simplex!(simplices, newindices)
    add_edge!(simplices, index0, index2)  # index0 is a parent of index2
    index1, index2
end

function best_worst_vertices(point_indices, heights)
    hmin = Inf
    hmax = -Inf
    jmin = 0
    jmax = 0
    for j in point_indices
        h = heights[j]
        if h < hmin
            hmin = h
            jmin = j
        end
        if h > hmax
            hmax = h
            jmax = j
        end
    end
    jmin, jmax
end

function push_new_simplex!(simplices, points)
    add_vertex!(simplices)
    newindex = nv(simplices)
    set_prop!(simplices, newindex, :points, points)
    push!(store[:notdone], newindex)
    newindex
end

"Integral of f over the ith simplex"
integral(simplices, index::Int) = get_prop(simplices, index, :integral)

"Sum the integrals of the component simplices (leaf nodes in the simplcies graph)."
function totalintegral!(simplices)
    val = 0.0
    nsimplices = nv(simplices)
    for i = 1:nsimplices
        !isempty(children(simplices, i)) && continue  # Simplex is not a leaf node
        val += integral(simplices, i)
    end
    val
end

function parent(simplices, childid)
    x = inneighbors(simplices, childid)
    length(x) == 0 && return 0
    length(x) == 1 && return x[1]
    error("Simplex has more than 1 parent")
end

children(simplices, parentid) = outneighbors(simplices, parentid)

end