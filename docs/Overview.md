# Overview

## Population

The population of interest, together with contact networks for households, schools, work places, social and community networks,
is constructed as described in [Demographics.jl](https://github.com/JockLawrie/Demographics.jl/blob/master/docs/Overview.md).

Note that contacts at schools and work places are currently constrained to occur only on weekdays.

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

## A Brief Note on Simulated Time

The model has a `date` property with a value of type `Date`, with unit 1 day.
Metrics reported on a given date specify the state of the system __as at 12am on that date__.
That is, a new date ticks over and the state of play is recorded before any new events take place.
All events that occur can be thought of as occuring between 12:00:01 and 23:59:59 inclusive.

For example, suppose a person is infected (state E) on 10th March (between 12:00:01 and 23:59:59 inclusive) and has an incubation period of 7 days.
On the 15th or 16th of March this person will become infectious but remain asymtpomatic (state IA).
On 17th March this person will be recorded with state IA, because at 12am the person hasn't yet transitioned to being symptomatic.
Then the person will become symptomatic (state IS) between 12:00:01 and 23:59:59, and recorded as such on the 18th March.

At the technical level, an event is a function together with some arguments and a scheduled time for execution.
When the function is executed, more events may be scheduled as part of the execution.
For example, when a person is tested for Covid, his/her _last_test_date_ is updated and test results are scheduled for 2 days into the future.
S/he is also quarantined until the test result is available, with compliance determined by the relevant model parameter.

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
   There are 6 such states.
   In order of increasing severity they are: asymptomatic, symptomatic without hospitalisation, ward bed, ICU, ventilation, death.

2. The sequence of states that a person will experience is determined by the most severe state experienced.
   Starting from exposure, the 6 sequences are:

   - Exposure -> Infectious-Asymptomatic -> Recovered
   - Exposure -> Infectious-Asymptomatic -> Infectious-Symptomatic -> Home -> Recovered
   - Exposure -> Infectious-Asymptomatic -> Infectious-Symptomatic -> Home -> Ward -> Recovered
   - Exposure -> Infectious-Asymptomatic -> Infectious-Symptomatic -> Home -> Ward -> ICU -> Ward -> Recovered
   - Exposure -> Infectious-Asymptomatic -> Infectious-Symptomatic -> Home -> Ward -> ICU -> Ventilation -> ICU -> Ward -> Recovered
   - Exposure -> Infectious-Asymptomatic -> Infectious-Symptomatic -> Home -> Ward -> ICU -> Ventilation -> Deceased

3. If asymptomatic, the person is considered recovered at the end of the infectious period.

4. If symptomatic the total duration of symptoms is drawn from a distribution estimated from DHHS data.
   Specifically, the distribution of total duration of symptoms is log-normal with location parameter a function of the patient's most severe state.
   In particular we have

       `Duration of symptoms | Patient becomes symptomatic ~ LogNormal(b0 + b1*Ward + b2*ICU + b3*Ventilation + b4*age, s)`

   where `Ward`, `ICU` and `Ventilation` are indicators (binary variables) of the patient's most severe state, and age is in years.
   The parameter `b0` determines the duration of symptoms for a symptomatic infant (age = 0) not requiring hospitalisation.

5. The total duration is then partitioned into segments corresponding to the sequence of states that the patient will experience.
   The partition is chosen according to a multinomial distribution estimated from the data.
   That is, we currently assume that everyone following a given sequence will have the same proportion of their symptomatic period in ICU (for example),
   even though these patients will have different durations for their symptomatic periods.
   Further investigation may warrant a different partitioning of the symptomatic period for each individual,
   in which case a Dirichlet distribution can be used.