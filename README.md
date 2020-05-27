# Covid.jl

## Run the code

```julia
repodir = "C:\\projects\\repos\\Covid.jl"
using Pkg
Pkg.activate(repodir)
using Covid
Covid.main(joinpath(repodir, "config", "config.yml"))
```

## States

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

The _actual_ number of new cases is E.
The _actual_ number of active cases is N - S - R - D.
The _actual_ number of cumulative cases is N - S.

## Contacts

Each person has the following contacts:

- Household: The people you live with.
- Social:    Family and friends that you don't live with.
- School:    People you see at school. Applicable only if you're 23 or under, or if your work place is a school.
- Workplace: People you work with. Applicable if you don't attend school and don't work at a school.
- Community: Strangers that you interact with when shopping, commuting, visiting the public library, cinema, etc.

## Time

The model has a `time` property with integer value.
It is the number of complete time periods that have elapsed.
Therefore it starts at 0.
It can also be considered the beginning of the period `[t, t+1)`.

The following occurs in order:

1. Set `T`, the number of complete time periods in the simulation, i.e., the number of steps to take. Therefore the simulation period is `[0, T]`.
2. Set the initial model state.
3. Set `t=0`, signifiying the start of the period `[0, 1)`.
4. Collect data on the state of the model as at `t` (which is the start of `[t, t+1)`). If `t==T`, STOP.
5. Make state changes. These can be considered to have occured during `(t, t+1)`.
6. Increment `t` by 1 unit. Go to Step 4.

Given this definition of time, if a state is scheduled to have duration 7 time units (starting at 0),
the state will change _after_ 7 time units have elapsed.
According to the above, when `t=7`:

1. Data is collected as at `t=7`.
2. The scheduled state change occurs during `(7, 8)`, i.e., after 7 periods but before 8 periods have elapsed.
3. Time is incremented to `t=8`.
4. Data is collected as at `t=8`. The state change is recorded as having _occurred by_ `t=8`, with the change actually occurring during `(7, 8)`.

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