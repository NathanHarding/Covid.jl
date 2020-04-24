module contacts

export populate_contacts!

using Distributions
using LightGraphs

function populate_contacts!(agents, indata, age2first)
    d_nparents  = Categorical([0.26, 0.74])
    d_nchildren = Categorical([0.33, 0.4, 0.25, 0.01, 0.005, 0.005])
    d_nadults_without_children = Categorical([0.42, 0.47, 0.04, 0.04, 0.02, 0.01])
    populate_households!(agents, age2first, d_nparents, d_nchildren, d_nadults_without_children)
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

function populate_households!(agents, age2first, d_nparents, d_nchildren, d_nadults_without_children)
    populate_households_with_children!(agents, age2first, d_nparents, d_nchildren)
    populate_households_without_children!(agents, age2first, d_nadults_without_children)
end

"""
- Input data: 
  - d_nparents: Pr(nadults == k | nchildren > 0) = [0.26, 0.74]
  - d_nchildren: Pr(nchildren == k | nchildren > 0). E.g., [0.33, 0.4, 0.25, 0.01, 0.005, 0.005]
while n_unplaced_children > 0
    sample nchildren household from d_nchildren
    sample nadults from d_nparents
    construct household
    populate household
       - kids no more than (nchildren - 1) x 3 years apart
       - adults at least 20 years older than oldest child
    set household contacts
"""
function populate_households_with_children!(agents, age2first, d_nparents, d_nchildren)
    unplaced_children = Set(1:(age2first[18] - 1))
    unplaced_parents  = Set((age2first[20]):(age2first[55] - 1))  # Parents of children under 18 are adults aged between 20 and 54
    imax = length(unplaced_children)
    for i = 1:imax  # Cap the number of iterations by placing at least 1 child per iteration
        # Init household
        n_unplaced_parents  = length(unplaced_parents)
        n_unplaced_children = length(unplaced_children)
        np = rand(d_nparents)
        nc = rand(d_nchildren)
        np = np > n_unplaced_parents  ? n_unplaced_parents  : np
        nc = nc > n_unplaced_children ? n_unplaced_children : nc
        hh = Household(np, nc)

        # Select children
        min_age = 0   # Minimum age of the next selected child
        max_age = 17  # Maximum age of the next selected child
        age_youngest_child = 1000
        age_oldest_child   = -1
        for i = 1:nc
            child_id = sample_person(unplaced_children, min_age, max_age, agents, age2first)
            pop!(unplaced_children, child_id)
            push_child!(hh, child_id)
            age = agents[child_id].age
            age_youngest_child = age < age_youngest_child ? age : age_youngest_child
            age_oldest_child   = age > age_oldest_child   ? age : age_oldest_child
            min_age = max(0,  age_youngest_child - 3 * (nc - 1))
            max_age = min(17, age_oldest_child   + 3 * (nc - 1))
        end

        # Select parent/s
        min_parent_age = age_oldest_child + 20
        max_parent_age = age_oldest_child + 45
        for i = 1:np
            parent_id = sample_person(unplaced_parents, min_parent_age, max_parent_age, agents, age2first)
            pop!(unplaced_parents, parent_id)
            push_adult!(hh, parent_id)
        end

        # Set household contacts
        set_household_contacts!(agents, hh)

        # Stopping criteria
        isempty(unplaced_children) && break
        isempty(unplaced_parents)  && break
    end
end

"""
- Input data:
  - d_nadults_without_children: Pr(nadults == k | nchildren == 0). E.g., Proportional to [0.24, 0.27, 0.02, 0.02, 0.01, 0.01].
while n_unplaced_adults > 0
    sample 1 household from d_nadults_without_children
    construct household
    populate household
        - No constraints at the moment
    set household contacts
"""
function populate_households_without_children!(agents, age2first, d_nadults_without_children)
    unplaced_adults = Set([agent.id for agent in agents if isempty(agent.household)])
    imax = length(unplaced_adults)
    for i = 1:imax  # Cap the number of iterations by placing at least 1 child per iteration
        # Init household
        n_unplaced_adults = length(unplaced_adults)
        na = rand(d_nadults_without_children)
        na = na > n_unplaced_adults ? n_unplaced_adults  : na
        hh = Household(na, 0)

        # Select adult/s
        for i = 1:na
            adult_id = rand(unplaced_adults)
            pop!(unplaced_adults, adult_id)
            push_adult!(hh, adult_id)
        end

        # Set household contacts
        set_household_contacts!(agents, hh)

        # Stopping criteria
        isempty(unplaced_adults) && break
    end
end

"Returns the id of an agent in the age range if one exists, else returns a random id from ids."
function sample_person(unplaced_people::Set{Int}, min_age, max_age, agents, age2first)
    i1 = age2first[min_age]
    i2 = age2first[max_age + 1] - 1
    s  = i1:i2      # Indices of people in the age range
    n  = length(s)  # Maximum number of random draws
    for i = 1:n
        id = rand(s)  # id is a random person in the age range
        id in unplaced_people && return id  # id is also an unplaced person
    end
    rand(unplaced_people)
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

#=
###################################################################
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
function attempt1_populate_households!(agents, d_household)
    nh_total = 0
    d_household[!, :n_households] = fill(0, size(d_household, 1))
    nonfull = Set{Household}()  # Partially-filled households
nadults = 0
    for agent in agents
        if agent.age <= 17
            nh_total = add_child_to_household!(agents, agent.id, nonfull, d_household, nh_total)
        else
nadults += 1
if nadults > 20_000 && nadults < 20_050
    println(agent.id, ".  ", length(nonfull))  # should be decreasing...but it's not. we have households without adults not getting filled.
else
    error("early finish")
end
            nh_total = add_adult_to_household!(agents, agent.id, nonfull, d_household, nh_total)
        end
    end

d_household[!, :p_actual] = d_household.n_households ./ nh_total
println(d_household[:, [2,3,4,5,6]])
println(length(nonfull))

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
        household = create_new_household_for_child(d_household, nh_total)  # Create household and update running proportion
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
        household = create_new_household_for_adult(d_household, nh_total)  # Create household and update running proportion
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

function create_new_household_for_child(d_household, nh_total)
    n = size(d_household, 1)
    for i = 1:n
        p_running  = d_household[i, :n_households] / nh_total
        p_running >= d_household[i, :proportion_of_households] && continue  # Target proportion has been met
        d_household[i, :nchildren] == 0 && continue  # No children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
    r = 1:n  # No suitable house...pick one at random
    for j = 1:1000
        i = rand(r)
        d_household[i, :nchildren] == 0 && continue  # No children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
end

function create_new_household_for_adult(d_household, nh_total)
    n = size(d_household, 1)
    for i = 1:n  # Households without children
        p_running  = d_household[i, :n_households] / nh_total
        p_running >= d_household[i, :proportion_of_households] && continue  # Target proportion has been met
        d_household[i, :nchildren] > 0 && continue  # Children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end

    #i=1  # ok
    #i = rand(1:n)  # takes too long
    #d_household[i, :n_households] += 1
    #return Household(d_household[i, :nadults], d_household[i, :nchildren])

    #=
    for i = 1:n  # Households with children
        p_running  = d_household[i, :n_households] / nh_total
        p_running >= d_household[i, :proportion_of_households] && continue  # Target proportion has been met
        #d_household[i, :nchildren] > 0 && continue  # Children in this household
        d_household[i, :n_households] += 1
        return Household(d_household[i, :nadults], d_household[i, :nchildren])
    end
    =#
    r = 1:n
    for j = 1:1000
        i = rand(r)  # No suitable house...pick one at random
        d_household[i, :nchildren] > 0 && continue  # Children in this household
        d_household[i, :n_households] += 1
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