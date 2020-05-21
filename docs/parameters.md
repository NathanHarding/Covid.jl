# Parameters

The __incubation__ period is the period between exposure on the onset of symptoms.
Using cases with known exposure and onset dates, the incubation period is distributed LogNormal(1.52, 0.58).
Note that the incubation period spans the Exposed (E) and Infectious-Asymptomatic (IA) stages of the model.
For patients who go on to be symptomatic:
- Draw an incubation duration from the LogNormal.
- Set Exposed duration = incubation duration - 2 days
- Set Asymptomatic duration = 2 days
For patients who remain asymptomatic:
- Draw an incubation duration from the LogNormal.
- Set Exposed duration = incubation duration - 2 days
- Set Asymptomatic duration = 2 days + draw from duration of symptomatic stage (IS)