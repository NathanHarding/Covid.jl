# Covid.jl

## Run the code

```julia
repodir = "C:\\projects\\repos\\Covid.jl"
using Pkg
Pkg.activate(repodir)
using Covid
Covid.main(joinpath(repodir, "config", "config.yml"))
```

## Population

The population of interest is constructed from ABS data, which gives an estimated population by age, sex and SA2.
To faciliate understanding of disease transmission and the effects of various control policies, people are allocated to households,
schools and work places. In addition, people have social and community contacts. See the __Contacts__ section below for details.

### Contacts

Each person has the following contacts:

- Household: The people you live with.
- Social:    Family and friends that you don't live with.
- School:    People you see at school. Applicable only if you're 23 or under, or if your work place is a school.
- Workplace: People you work with. Applicable if you don't attend school and don't work at a school.
- Community: Strangers that you interact with when shopping, commuting, visiting the public library, cinema, etc.

Note that contacts at schools and work places only take place on weekdays.

Each member of the population is then allocated to a household as follows:

1. Household data from the ABS gives an estimated count of households by household size, and also family composition.
   We use this data to construct a simplified set households consisting of households with children and 1 or 2 parents,
   and households without children.
2. Households with children are constucted by:
   - Randomly drawing a household from the set of constructed households (with the numbers of adults and children determined)
   - Randomly draw the children such that they are no more than 5 years apart
   - Randomly draw the parent or parents such that they are no more than 45 years older than the oldest child
3. Households without children are then populated using random allocation of remaining adults.

Schools are constructed from DET data that gives the number of children in each year level for individual schools.
Children aged 5-17 are allocated to schools at random with their age matched to their year level.
Teachers are allocated according to a fixed teacher:student ratio of 1:15.
Contacts among children are constructed using a regular graph - each member has the same number of contacts, thus inducing many common contacts between 2 neighouring students in the graph, but also some different contacts.
Contacts among teachers are also constructed as regular graphs.
Contacts between students and teachers are constructed by teachers having a fixed number of student contacts.

Child care centres are constructed for children aged 0 to 4 in a similar way.
Since we lack data we assume that a child care centre has a room for each age group containing 20 children.

Universities and TAFES are constructed similarly for adults aged 18 to 23.
We assume 1000 students per age group and a teacher:student ratio of 1:40.

Work places are constructed from ABS data concerning the number of employees and populated randomly with remaining adults.
Contacts within work places are constructed as regular graphs.

Social and community contacts are regular within the entire population.

## States

The set of possible states for each person in the population is:

- S:   Susceptible. Not infected (next state is E).
- E:   Exposed. Infected but asymptomatic and not infectious (next state is IA).
- IA:  Infectious and asymptomatic (next state is IS at the end of the incubation period, or R at the end of the infectious period).
- IS:  Infectious and symptomatic (next state is H at the end of the infectious period).
- H:   Home and symptomatic but not infectious (next state is W or R).
- W:   Admitted to a ward bed (next state is ICU or R).
- ICU: Admitted to an unventilated ICU bed (next state is V or W).
- V:   On a ventilator in ICU (next state is D or ICU).
- R:   Recovered (final state).
- D:   Deceased (final state).

### A Brief Note on Time

The model has a `date` property with a value of type `Date`, with unit 1 day.
Metrics reported on a given date specify the state of the system as at 12am on that date.
That is, a new date ticks over and the state of play is recorded before any new events take place.
All events that occur can be thought of as occuring between 12:00:01 and 23:59:59 inclusive.

For example, suppose a person is infected (state E) on 10th March (between 12:00:01 and 23:59:59 inclusive) and has an incubation period of 7 days.
On the 15th or 16th of March this person will become infectious but remain asymtpomatic (state IA).
On 17th March this person will be recorded with state IA, because at 12am the person hasn't yet transitioned to being symptomatic.
Then the person will become symptomatic (state IS) between 12:00:01 and 23:59:59, and recorded as such on the 18th March.

At the technical level, an event is a function together with some arguments and a scheduled time for execution.
When the function is executed, more events may be scheduled as part of the execution.

For example, when a person is tested for Covid, his/her _last_test_date_ is updated and test results are scheduled for 2 days into the future.
S/he is also quarantined until the test result is available, with compliance being a model parameter.

## The Sequence of Events

Each simulation begins with the state of each person pre-specified.
At each time step (i.e., on each day), infectious individuals make contact with members of their contact lists as specified in the configuration.

When an infectious person makes contact with a susceptible person, the contact is infected with probabililty specified in the model parameters.

On being infected a person transitions from susecptible (state is S) to exposed (state is E).
The incubation period is determined by drawing from a log-normal distribution consistent with the average and decay described by the DHHS Covid literature review team.
One or two days prior to the end of the incubation period the person becomes infectious while remaining asymptomatic (state is IA).
At the end of the incubation period the person either becomes symptomatic (state is IS) or remains asymptomatic, with probability estimated from the literature.

From the end of the infectious period the sequence of events that determine the person's entire trajectory are determined and scheduled as follows:

1. The most severe state that the person will experience is determined by drawing from a distribution that depends on his/her risk factors (currently just age).
   There are 6 such states, namely asymptomatic, symptomatic without hospitalisation, ward bed, ICU, ventilation, death.
2. If asymptomatic, the person is considered recovered at the end of the infectious period.
3. If symptomatic the total duration of symptoms is drawn from a distribution estimated from DHHS data.
   Specifically, the distribution of total duration of symptoms is log-normal with location parameter a function of the patient's most severe state.
   In particular we have

       `Duration of symptoms | Patient becomes symptomatic ~ LogNormal(b0 + b1*Ward + b2*ICU + b3*Ventilation + b4*age, s)`

   where `Ward`, `ICU` and `Ventilation` are indicators (binary variables) of the patient's most severe state, and age is in years.
   The parameter `b0` determines the duration of symptoms for a symptomatic infant (age = 0) not requiring hospitalisation.

4. The total duration is then partitioned into segments corresponding to the sequence of states that the patient will experience.
   For example, a patient whose most severe state is ventilation will follow this sequence:

   `Home -> Ward -> ICU -> Ventilation -> ICU -> Ward -> Recovery`

   The partition is chosen according to a multinomial distribution estimated from the data.
   That is, we currently assume that everyone following a given sequence will have the same proportion of their symptomatic period in ICU (for example),
   even though these patients will have different durations for their symptomatic periods.
   Further investigation may warrant a different partitioning of the symptomatic period for each individual,
   in which case a Dirichlet distribution can be used.

## Policies

### Physical Distancing

### Testing

### Tracing

### Quarantining


## Training the model

We estimate parameters for transition probabilities and conditional duration distributions from the literature and from DHHS linked data.

We estimate the probabilities that make up the baseline policies by training the model for the period 25th Jan to 23rd Mar inclusive.
This baseline period includes restrictions on overseas arrivals, but otherwise excludes Stage 1/2/3 restrictions.
On the other hand many people were working from home and/or self-isolating before 23/3.


## Next

- Documentation
- Calibrate parameters and initial values.
- Allow for interstate and international arrivals.
- Parition population for specific hospitals
    - Use SA4 for workplace distribution, assuming people can travel across their SA4 of residence to work
- Add other risk factors such as diabetic status, hypertension, BMI, etc.