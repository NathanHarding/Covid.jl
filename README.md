# Covid.jl

# Run the code

```julia
cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")
using Covid
configfile = "config\\config.yml"
Covid.main(configfile)
```

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

- Update parameters with ABS data.
  - Update schools data
  - Read Simon's work, compare to the household code
- Fix scheduling for infecting contacts
- Define social distancing scenarios.
- Solve for parameters.
- Parition population for specific hospitals
    - Use SA4 for workplace distribution, assuming people can travel across their SA4 of residence to work
- Add other risk factors such as diabetic status, hypertension, BMI, etc.