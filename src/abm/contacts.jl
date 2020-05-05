module contacts

export populate_contacts!

using DataFrames
using Distributions
using LightGraphs

function populate_contacts!(agents, cfg, indata)
    # household data
    d_nparents  = Categorical([0.26, 0.74])
    d_nchildren = Categorical([0.33, 0.4, 0.25, 0.01, 0.005, 0.005])
    d_nadults_without_children = Categorical([0.42, 0.47, 0.04, 0.04, 0.02, 0.01])

    age2first = construct_age2firstindex!(agents)  # agents[age2first[i]] is the first agent with age i
    populate_households!(agents, age2first, d_nparents, d_nchildren, d_nadults_without_children)
    populate_school_contacts!(agents, age2first, indata["school_distribution"], cfg.ncontacts_s2s, cfg.ncontacts_t2t, cfg.ncontacts_t2s)
    populate_workplace_contacts!(agents, cfg.n_workplace_contacts, indata["workplace_distribution"])
    populate_community_contacts!(agents, cfg.n_community_contacts)
    populate_social_contacts!(agents, cfg.n_social_contacts)
end

################################################################################
# Utils

function append_contact!(agentid, contactid::Int, contactlist::Vector{Int})
    agentid == contactid && return
    push!(contactlist, contactid)
end

"Append contactids to agent.category"
function append_contacts!(agent, category::Symbol, contactids::Vector{Int})
    agentid = agent.id
    contactlist = getproperty(agent, category)  # Vector{Int}
    for contactid in contactids
        append_contact!(agentid, contactid, contactlist)
    end
end

"Randomly assign ncontacts to each agent in agents."
function assign_contacts_regulargraph!(agents, contactcategory::Symbol, ncontacts::Int)
    npeople   = length(agents)
    ncontacts = adjust_ncontacts_for_regular_graph(npeople, ncontacts)  # Ensure a regular graph can be constructed
    g         = random_regular_graph(npeople, ncontacts)  # npeople (vertices) each with ncontacts (edges to ncontacts other vertices)
    adjlist   = g.fadjlist
    for agent in agents
        setproperty!(agent, contactcategory, adjlist[agent.id])
    end
end

"Randomly assign ncontacts to each agent whose id is a value of vertexid2agentid."
function assign_contacts_regulargraph!(agents, contactcategory::Symbol, ncontacts::Int, vertexid2agentid)
    nvertices = length(vertexid2agentid)
    ncontacts = adjust_ncontacts_for_regular_graph(nvertices, ncontacts)  # Ensure a regular graph can be constructed
    g = random_regular_graph(nvertices, ncontacts)  # nvertices each with ncontacts (edges to ncontacts other vertices)
    adjlist = g.fadjlist
    for (vertexid, agentid) in vertexid2agentid
        contactlist_vertex = adjlist[vertexid]  # Contact list as vertexid domain...convert to agentid domain
        contactlist_agent  = getproperty(agents[agentid], contactcategory)
        for vertexid in contactlist_vertex
            append_contact!(agentid, vertexid2agentid[vertexid], contactlist_agent)
        end
    end
end

"""
Require these conditions:
1. nvertices >= 1
2. ncontacts <= nvertices - 1
3. iseven(nvertices * ncontacts)

If required adjust ncontacts.
Return ncontacts.
"""
function adjust_ncontacts_for_regular_graph(nvertices, ncontacts)
    nvertices == 0 && error("The number of vertices must be at least 1")
    ncontacts = ncontacts > nvertices - 1 ? nvertices - 1 : ncontacts
    iseven(nvertices * ncontacts) ? ncontacts : ncontacts - 1
end

"""
Return the id of a random agent whose is unplaced and in the specified age range.
If one doesn't exist return a random id from unplaced_people.
"""
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

"""
- Sort agents
- Rewrite their ids in order
- Return age2first, where agents[age2first[i]] is the first agent with age i
"""
function construct_age2firstindex!(agents)
    sort!(agents, by=(x) -> x.age)  # Sort from youngest to oldest
    age2first = Dict{Int, Int}()    # age => first index containing age
    current_age = -1
    for i = 1:length(agents)
        agent = agents[i]
        agent.id = i
        age = agent.age
        if age != current_age
            current_age = age
            age2first[current_age] = i
        end
    end
    age2first
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
    hh2count = Dict{Tuple{Int, Int}, Int}()  # (nadults, nchildren) => count(households). For evaluation only.
    populate_households_with_children!(agents, age2first, hh2count, d_nparents, d_nchildren)
    populate_households_without_children!(agents, age2first, hh2count, d_nadults_without_children)
#=
# Evaluation code
nh = sum(collect(values(hh2count)))
for (k, v) in hh2count
    p = 100 * round(v/nh; digits=3)
    println("$(k)  => $(v),   $(p)%")
end
println(nh)
error("early finish")
=#
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
function populate_households_with_children!(agents, age2first, hh2count, d_nparents, d_nchildren)
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
        for j = 1:nc
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
        for j = 1:np
            parent_id = sample_person(unplaced_parents, min_parent_age, max_parent_age, agents, age2first)
            pop!(unplaced_parents, parent_id)
            push_adult!(hh, parent_id)
        end

        # Set household contacts
        set_household_contacts!(agents, hh)
        hh2count[(np, nc)] = haskey(hh2count, (np, nc)) ? hh2count[(np, nc)] + 1 : 1

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
function populate_households_without_children!(agents, age2first, hh2count, d_nadults_without_children)
    unplaced_adults = Set([agent.id for agent in agents if isempty(agent.household)])
    imax = length(unplaced_adults)
    for i = 1:imax  # Cap the number of iterations by placing at least 1 child per iteration
        # Init household
        n_unplaced_adults = length(unplaced_adults)
        na = rand(d_nadults_without_children)
        na = na > n_unplaced_adults ? n_unplaced_adults  : na
        hh = Household(na, 0)

        # Select adult/s
        for j = 1:na
            adult_id = rand(unplaced_adults)
            pop!(unplaced_adults, adult_id)
            push_adult!(hh, adult_id)
        end

        # Set household contacts
        set_household_contacts!(agents, hh)
        hh2count[(na, 0)] = haskey(hh2count, (na, 0)) ? hh2count[(na, 0)] + 1 : 1

        # Stopping criteria
        isempty(unplaced_adults) && break
    end
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

################################################################################
# School contacts for teachers and people under 23

"""
Each teacher can contact N1 other teachers and N2 students from any age group.
Each student can contact N3 students in their age group.
"""
mutable struct School
    max_nteachers::Int
    max_nstudents_per_level::Int
    teachers::Vector{Int}
    age2students::Dict{Int, Vector{Int}}  # Classes: age => [childid1, ...]

    function School(max_nteachers, max_nstudents_per_level, teachers, age2students)
        teacher2student_ratio = 1 / 15  # Need at least 1 teacher to 15 students
        nlevels       = length(age2students)
        max_nstudents = nlevels * max_nstudents_per_level
        min_nteachers = round(Int, teacher2student_ratio * max_nstudents)
        max_nteachers < min_nteachers && error("Not enough teachers")  # Need 1.5 per level, 6 levels
        max_nstudents_per_level < 5    && error("max_nstudents_per_level must be at least 5")
        new(max_nteachers, max_nstudents_per_level, teachers, age2students)
    end
end

function School(schooltype::Symbol, max_nstudents_per_level)
    # Construct age2students
    if schooltype == :childcare
        age2students = Dict(age => Int[] for age = 0:4)
    elseif schooltype == :primary
        age2students = Dict(age => Int[] for age = 5:11)
    elseif schooltype == :secondary
        age2students = Dict(age => Int[] for age = 12:17)
    elseif schooltype == :tertiary
        age2students = Dict(age => Int[] for age = 18:23)
    else
        error("Unknown school type")
    end

    # Calculate the number of required teachers
    teacher2student_ratio = 1 / 15  # Need at least 1 teacher to 15 students
    nlevels       = length(age2students)
    max_nstudents = nlevels * max_nstudents_per_level
    max_nteachers = round(Int, teacher2student_ratio * max_nstudents)
    School(max_nteachers, max_nstudents_per_level, Int[], age2students)
end

function isfull(school::School)
    length(school.teachers) >= school.max_nteachers && return true
    for (age, students) in school.age2students
        length(students) < school.max_nstudents_per_level && return false  # Room in this age group
    end
    true
end

function push_teacher!(school::School, id::Int)
    length(school.teachers) >= school.max_nteachers && return false  # Teacher positions are full. No success.
    push!(school.teachers, id)  # Add teacher to school
    true  # Success
end

function push_student!(school::School, id::Int, age::Int)
    v = school.age2students[age]
    length(v) >= school.max_nstudents_per_level && return false  # Child positions are full. No success.
    push!(v, id)  # Add teacher to school
    true  # Success
end

function populate_school_contacts!(agents, age2first, school_distribution::DataFrame, ncontacts_s2s, ncontacts_t2t, ncontacts_t2s)
    d_nstudents_per_level = Categorical(school_distribution.proportion)
    min_teacher_age   = 24
    max_teacher_age   = 65
    unplaced_students = Dict(age => Set(age2first[age]:(age2first[age+1] - 1)) for age = 0:23)
    unplaced_teachers = Set((age2first[min_teacher_age]):(age2first[max_teacher_age] - 1))
    imax              = length(unplaced_students)
    for i = 1:imax  # Cap the number of iterations by placing at least 1 child per iteration
        # Init school
        agentid = nothing
        for age = 0:23
            isempty(unplaced_students[age]) && continue
            agentid = rand(unplaced_students[age])
            break
        end
        isnothing(agentid) && break  # STOPPING CRITERION: There are no unplaced students remaining
        student    = agents[agentid]
        schooltype = determine_schooltype(student.age)
        max_nstudents_per_level = draw_nstudents_per_level(school_distribution, d_nstudents_per_level)
        school     = School(schooltype, max_nstudents_per_level)

        # Fill student positions
        age2students = school.age2students
        for (age, students) in age2students
            n_available = max_nstudents_per_level - length(students)  # Number of available positions
            for j = 1:n_available
                isempty(unplaced_students[age]) && break
                studentid = sample_person(unplaced_students[age], age, age, agents, age2first)
                pop!(unplaced_students[age], studentid)
                push_student!(school, studentid, age)
            end
        end

        # Fill teacher positions
        n_available = school.max_nteachers - length(school.teachers)  # Number of available positions
        for j = 1:n_available
            isempty(unplaced_teachers) && break
            teacherid = sample_person(unplaced_teachers, min_teacher_age, max_teacher_age, agents, age2first)
            pop!(unplaced_teachers, teacherid)
            push_teacher!(school, teacherid)
        end

        # Set contact lists
        set_student_to_student_contacts!(agents, school, ncontacts_s2s)
        set_teacher_to_student_contacts!(agents, school, ncontacts_t2s)
        set_teacher_to_teacher_contacts!(agents, school, ncontacts_t2t)
    end
end

function draw_nstudents_per_level(school_distribution::DataFrame, d_nstudents_per_level)
    i  = rand(d_nstudents_per_level)
    lb = school_distribution[i, :avg_year_level_size_lb]
    ub = school_distribution[i, :avg_year_level_size_ub]
    round(Int, 0.5 * (lb + ub))
end

function determine_schooltype(age::Int)
    age <= 4  && return :childcare
    age <= 11 && return :primary
    age <= 17 && return :secondary
    age <= 23 && return :tertiary
    error("Person with age $(age) cannot be assigned as a student to a school")
end

function set_student_to_student_contacts!(agents, school::School, ncontacts_s2s::Int)
    age2students = school.age2students
    for (age, students) in age2students
        isempty(students) && continue
        nstudents        = length(students)
        vertexid2agentid = Dict(i => students[i] for i = 1:nstudents)
        assign_contacts_regulargraph!(agents, :workplace, min(ncontacts_s2s, nstudents), vertexid2agentid)
    end
end

function set_teacher_to_student_contacts!(agents, school::School, ncontacts_t2s::Int)
    # Construct a vector of studentids
    studentids = Int[]
    for (age, students) in school.age2students
        for studentid in students
            push!(studentids, studentid)
        end
    end
    nstudents = length(studentids)

    # For each teacher, cycle through the students until the teacher has enough contacts
    teachers = school.teachers
    ncontacts_t2s = min(ncontacts_t2s, nstudents)  # Can't contact more students than are in the school
    idx = 0
    for teacherid in teachers
        teacher_contactlist = agents[teacherid].workplace
        for i = 1:ncontacts_t2s
            idx += 1
            idx  = idx > nstudents ? 1 : idx
            studentid = studentids[idx]
            student_contactlist = agents[studentid].workplace
            append_contact!(teacherid, studentid, teacher_contactlist)
            append_contact!(studentid, teacherid, student_contactlist)
        end
    end
end

function set_teacher_to_teacher_contacts!(agents, school::School, ncontacts_t2t::Int)
    teachers = school.teachers
    isempty(teachers) && return
    nteachers        = length(teachers)
    vertexid2agentid = Dict(i => teachers[i] for i = 1:nteachers)
    assign_contacts_regulargraph!(agents, :workplace, min(ncontacts_t2t, nteachers), vertexid2agentid)
end

################################################################################
# Workplace, community and social contacts

function populate_workplace_contacts!(agents, ncontacts::Int, workplaces::DataFrame)
    nagents = size(agents, 1)
    adultid = 0
    d_workplace_size = Categorical(workplaces.proportion)  # Categories are: 0 employees, 1-4, 5-19, 20-199, 200+
    workplace_size   = draw_nworkers(workplaces, d_workplace_size)
    adultid2agentid  = Dict{Int, Int}()
    for agent in agents
        agent.age <= 23               && continue  # People under 23 are assumed to be in education not the workplace
        !isempty(agent.workplace) > 0 && continue  # Adult is employed at a school
        adultid += 1
        adultid2agentid[adultid] = agent.id
        if adultid == workplace_size  # Work place is full...record the workplace contacts and set a new empty workplace.
            assign_contacts_regulargraph!(agents, :workplace, min(ncontacts, workplace_size), adultid2agentid)
            adultid = 0
            adultid2agentid = Dict{Int, Int}()
            workplace_size  = draw_nworkers(workplaces, d_workplace_size)
        end
    end
end

function draw_nworkers(workplaces::DataFrame, d_workplace_size)
    i  = rand(d_workplace_size)
    lb = workplaces[i, :nemployees_lb]
    ub = workplaces[i, :nemployees_ub]
    rand(lb:ub) + 1
end


populate_community_contacts!(agents, ncontacts) = assign_contacts_regulargraph!(agents, :community, ncontacts)
populate_social_contacts!(agents, ncontacts)    = assign_contacts_regulargraph!(agents, :social,    ncontacts)

end