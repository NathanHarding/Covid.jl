# Policies

There are 4 policy categories, outlined below.
Policies can change over time - a policy begins on a specified date and continues until another policy replaces it.

## Physical Distancing

The physical distancing policy specifies the probability of a person physically contacting a person in his/her contact list on a given day.
School and work place contacts are only made on weekdays.
A separate probability is specified for each contact type.

For example:

```yaml
distancing_policy:
    2020-01-25: {household: 0.5, school: 0.2,   workplace: 0.2, community: 0.1, social: 0.1}  # Normal conditions
    2020-03-24: {household: 0.5, school: 0.001, workplace: 0.2, community: 0.1, social: 0.1}  # School closures
```

## Testing

The testing policy specifies the probability of testing someone who is not a contact of a known case.
The probability depends on the person's state.

For example, the following policy tests infectious and symptomatic people with probability 0.3, as well as all people admitted to hospital.

```yaml
testing_policy:
    2020-01-25: {IS: 0.3, W: 1.0}
    2020-05-04: {IS: 0.8, W: 1.0}  # Increase testing intensity outside hospitals
```

## Tracing

The contact tracing policy specifies the probability of testing someone who is a contact of a known case.
The probability depends on the person's state.
Asymptomatic contacts are S, E, IA or R.
In this context the only symptomatic contacts who are traced have state IS (Infectious-Symptomatic),
since contacts who have been admitted to hospital have already been tested upon admission.

The example below shows that all household contacts are traced, as well as most symptomatic work place contacts.

```yaml
tracing_policy:
    2020-01-25: {household: {symptomatic: 1.0, asymptomatic: 1.0}, school: {symptomatic: 0.0, asymptomatic: 0.0},
                 workplace: {symptomatic: 0.8, asymptomatic: 0.0}, social: {symptomatic: 0.0, asymptomatic: 0.0},
                 community: {symptomatic: 0.0, asymptomatic: 0.0}}
```

## Quarantine

The quarantine policy specifies how many days people are quarantined, together with the level of compliance (expressed as a probability).
People who are awaiting test results are quarantined until the result is available, which is assumed to be 2 days after the test date.
People who test positive are quarantined for X days after onset of symptoms if they are symptomatic (status=IS), or X days after the test date if asymptomatic.
People known to be in recent contact with a known case are quarantined for X days.

```yaml
quarantine_policy:
    2020-01-25: {awaiting_test_result: {days: 2, compliance: 0.7},
                 tested_positive: {days: 10, compliance: 0.7},
                 case_contacts: {household: {days: 14, compliance: 0.7}}}
    2020-03-15: {awaiting_test_result: {days: 2, compliance: 0.7},
                 tested_positive: {days: 10, compliance: 0.7},
                 case_contacts: {household: {days: 14, compliance: 0.7}, workplace: {days: 14, compliance: 0.7}}}
```
