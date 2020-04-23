module contactnetworks

export populate_contacts!

using LightGraphs

function populate_contacts!(agents, indata)
    populate_households!(agents, indata["household_distribution"])
    populate_social_contacts!(agents)
end

function append_contact!(agent, category::Symbol, id::Int)
    id == agent.id && return  # A person is not a contact of him/herself
    v = getproperty(agent, category)  # Vector{Int}
    push!(v, id)
end

function append_contacts!(agent, category::Symbol, ids::Vector{Int})
    v = getproperty(agent, category)  # Vector{Int}
    ref_id = agent.id
    for id in ids
        id == ref_id && continue  # A person is not a contact of him/herself
        push!(v, id)
    end
end

################################################################################
# Households

struct Household
    max_nadults::Int
    max_nchildren::Int
    adults::Vector{Int}    # ids
    children::Vector{Int}  # ids

    function Household(max_nadults, max_nchildren, adults, children)
        length(adults)   > max_nadults        && error("Household has too many adults")
        length(children) > max_nchildren      && error("Household has too many children")
        max_nadults < 0                       && error("Household must have a non-negative number of adults")
        max_nchildren < 0                     && error("Household must have a non-negative number of children")
        max_nadults + max_nchildren == 0      && error("Household must have at least 1 resident")
        max_nadults == 0 && max_nchildren > 0 && error("Household with children must have at least 1 adult")
        new(max_nadults, max_nchildren, adults, children)
    end
end

Household(max_nadults, max_nchildren) = Household(max_nadults, max_nchildren, Int[], Int[])

isfull(hh::Household) = (length(hh.adults) == hh.max_nadults) && (length(hh.children) == hh.max_nchildren)

"Attempts to add an adult to the household. Returns true if successful."
function push_adult!(hh::Household, id)
    length(hh.adults) >= hh.max_nadults && return false  # Household is full. No success.
    push!(hh.adults, id)  # Add person to household
    true  # Success
end

"Attempts to add a child to the household. Returns true if successful."
function push_child!(hh::Household, id)
    length(hh.children) >= hh.max_nchildren && return false  # Household is full. No success.
    push!(hh.children, id)  # Add person to household
    true  # Success
end

"""
d_household: A DataFrame with columns ABS_category, nadults, nchildren, proportion_of_households.

1. Init:
   - For each row in d_household, set running proportion to 0.
   - Create an empty set of non-full households.
2. For each agent:
   - Fit into a non-full household if possible.
   - Otherwise:
     - Create an empty household:
         - With type that currently has running proportion less than the target proportion, and
         - That the person can go into.
     - Update the running proportion for the household type.
     - Append the person to the household.
     - If the household is not full, append the household to the set of non-full households.
   - If household is full, remove from the set of non-full households (if it's there) and update household contacts for all household members.
"""
function populate_households!(agents, d_household)
    nh_total = 0
    d_household[!, :n_households] = fill(0, size(d_household, 1))
    nonfull = Set{Household}()  # Partially-filled households
    for agent in agents
        if agent.age <= 17
            nh_total = add_child_to_household!(agents, agent.id, nonfull, d_household, nh_total)
        else
            nh_total = add_adult_to_household!(agents, agent.id, nonfull, d_household, nh_total)
        end
    end

#d_household[!, :p_actual] = d_household.n_households ./ nh_total
#println(d_household[:, [2,3,4,5,6]])

    for household in nonfull  # Remaining partially-filled households
        set_household_contacts!(agents, household)
    end
end

function add_child_to_household!(agents, id, nonfull, d_household, nh_total)
    household = nothing
    for hh in nonfull
        ok = push_child!(hh, id)
        !ok && continue
        household = hh
        break
    end
    if isnothing(household)  # A suitable household is not in nonfull
        household = create_new_household_with_child(d_household, nh_total)  # Create household and update running proportion
        nh_total += 1
        push_child!(household, id)
        if isfull(household)
            set_household_contacts!(agents, household)
        else
            push!(nonfull, household)
        end
    elseif isfull(household)  # household is in nonfull
        set_household_contacts!(agents, household)
        pop!(nonfull, household)
    end
    nh_total
end

function add_adult_to_household!(agents, id, nonfull, d_household, nh_total)
    household = nothing
    for hh in nonfull
        ok = push_adult!(hh, id)
        !ok && continue
        household = hh
        break
    end
    if isnothing(household)  # A suitable household is not in nonfull
        household = create_new_household_without_child(d_household, nh_total)  # Create household and update running proportion
        nh_total += 1
        push_adult!(household, id)
        if isfull(household)
            set_household_contacts!(agents, household)
        else
            push!(nonfull, household)
        end
    elseif isfull(household)  # household is in nonfull
        set_household_contacts!(agents, household)
        pop!(nonfull, household)
    end
    nh_total
end

function set_household_contacts!(agents, household)
    adults   = household.adults
    children = household.children
    for adult in adults
        append_contacts!(agents[adult], :household, adults)
        append_contacts!(agents[adult], :household, children)
    end
    for child in children
        append_contacts!(agents[child], :household, adults)
        append_contacts!(agents[child], :household, children)
    end
end

function create_new_household_with_child(d_household, nh_total)
    n = size(d_household, 1)
    for i = 1:n
        p_running  = d_household[i, :n_households] / nh_total
        p_running >= d_household[i, :proportion_of_households] && continue  # Target proportion has been met
        d_household[i, :nchildren] == 0 && continue  # No children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
    for i = 1:n  # If no household type has been found, ignore the target constraint
        d_household[i, :nchildren] == 0 && continue  # No children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
end

function create_new_household_without_child(d_household, nh_total)
    n = size(d_household, 1)
    for i = 1:n
        p_running  = d_household[i, :n_households] / nh_total
        p_running >= d_household[i, :proportion_of_households] && continue  # Target proportion has been met
        d_household[i, :nchildren] > 0 && continue  # Children in this household
        d_household[i, :n_households] += 1
        nh_total += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
    for i = 1:n  # If no household type has been found, ignore the target constraint
        d_household[i, :nchildren] > 0 && continue  # Children in this household
        d_household[i, :n_households] += 1
        nh_total += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
end

#=
"""
Appends a column containing the number of each type of household.

The counts are such that the total number of residents is npeople.    
"""
function append_nhouseholds!(d_households, npeople)
    n = size(d_households, 1)
    avg_residents_per_household = 0.0
    for i = 1:n
        avg_residents_per_household += (d[i, :nadults] + d[i, :nchildren]) * d[i, :proportion_of_households]
    end
    n_households = npeople / avg_residents_per_household
    d_households[!, :nhouseholds] = fill(0, n)
    for i = 1:n
        d_households[i, :nhouseholds] = round(Int, n_households * d[i, :proportion_of_households])
    end
end
=#

################################################################################
# Social contacts

function populate_social_contacts!(agents)
    npeople   = length(agents)
    ncontacts = 35
    g = random_regular_graph(npeople, ncontacts)  # npeople (vertices) each with ncontacts (edges to ncontacts other vertices)
    adjlist = g.fadjlist
    for id = 1:npeople
        agents[id].social = adjlist[id]
    end
end

end