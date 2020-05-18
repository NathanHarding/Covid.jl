# Covid.jl

## Run the code

```julia
repodir = "C:\\projects\\repos\\Covid.jl"
#repodir = "C:\\Users\\jlaw1812\\repos\\Covid.jl"
using Pkg
Pkg.activate(repodir)
using Covid
Covid.main(joinpath(repodir, "config", "config.yml"))
```

## States

- S:   Susceptible. Not infected (next state is E).
- E:   Exposed. Infected but asymptomatic and not infectious (next state is IA).
- IA:  Infectious and asymptomatic (next state is IS or R).
- IS:  Infectious and symptomatic (next state is H or R).
- W:   Admitted to a ward bed (next state is ICU or R).
- ICU: Admitted to an unventilated ICU bed (next state is V or W).
- V:   On a ventilator in ICU (next state is D or ICU).
- R:   Recovered (final state).
- D:   Deceased (final state).

The _actual_ number of new cases is E.
The _actual_ number of active cases is N - S - R - D.
The _actual_ number of cumulative cases is N - S.

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

## Next

- Testing regimes
  - Hospital admissions only
  - A proportion of symptomatic cases
  - A proportion of asymptomatic people (S, E, IA)
- __Contact tracing__ regimes
  - None
  - Test all contacts of a known case
  - Test all contacts of a tested case
- With probability p, p=0.5, 0.6, ..., 1.0, __isolate__ people who:
   - Take a test
   - Test positive
   - Are a contact of a tested person
   - Are a contact of a known case
   - Are in the same school/workplace as a known case (i.e., even if not in direct contact)
- Allow for interstate and international arrivals.
- Set initial values.
- Separate night and day, weekdays and weekends...period of 14 x 12-hour blocks.
- Define social distancing scenarios.
- Solve for parameters.
- Parition population for specific hospitals
    - Use SA4 for workplace distribution, assuming people can travel across their SA4 of residence to work
- Add other risk factors such as diabetic status, hypertension, BMI, etc.