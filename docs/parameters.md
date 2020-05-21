# Parameters

Here we describe the rationale for parameter choices.
The _modelling team_ refers to the DHHS Covid modelling team.
The _literature review team_ refers to the DHHS Covid literature review team.


After E, infectious duration ~ LogNormal(mu, sigma) such that mean = std = 1.

## Incubation Duration

The incubation period is the period between exposure and the onset of symptoms.
It spans the Exposed (E) and Infectious-Asymptomatic (IA) stages of the model.

Using cases with known exposure and onset dates, the modelling team estimated the incubation period as being distributed LogNormal(1.52, 0.58).
The mean of this distribution is 5.41 days, which is consistent with the estimate of 5.5 days published by the WHO-China joint mission on Coronavirus.

Also, the literature review team estimates that viral shedding begins about 2 days prior to symptoms, and reduces dramatically about 5 days after symptom onset.

So we have:

    incubation duration = exposed duration + IA duration
    exposed duration    = incubation duration - IA duration

indicating that the exposed duration is the incubation period minus .
For patients who go on to be symptomatic:
- Draw an incubation duration from the LogNormal.
- Set Exposed duration = incubation duration - 2 days
- Set Asymptomatic duration = 2 days
For patients who remain asymptomatic:
- Draw an incubation duration from the LogNormal.
- Set Exposed duration = incubation duration - 2 days
- Set Asymptomatic duration = 2 days + draw from duration of symptomatic stage (IS)