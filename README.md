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

- Susceptible: Not infected (next state is E)
- Exposed: Infected but asymptomatic and not infectious (next state is I1)
- I1: Infectious and asymptomatic (next state is I2 or R)
- I2: Infectious and symptomatic (next state is H or R).
  Assume a proportion __alpha__ of I2 cases get tested.
- Hospitalised: Admitted to a ward bed (next state is C or R)
- ICU: Admitted to ICU but not on a ventilator (next state is V or R)
- Ventilated: On a ventilator in ICU (next state is D or R)
- Recovered (final state)
- Deceased (final state)

The _actual_ number of new cases is E.
The _actual_ number of active cases is N - S - R - D.
The _actual_ number of cumulative cases is N - S.
The _observed_ number of new cases is alpha*new_I2_cases.
The _observed_ number of active cases is alpha*I2 + H + ICU + V.

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

- Read Simon's work, compare to the household code.
- Fix Pr(1 parent | children)
- Allow for interstate and international arrivals.
- Set initial values.
- Separate night and day, weekdays and weekends.
- Define social distancing scenarios.
- Solve for parameters.
- Parition population for specific hospitals
    - Use SA4 for workplace distribution, assuming people can travel across their SA4 of residence to work
- Add other risk factors such as diabetic status, hypertension, BMI, etc.