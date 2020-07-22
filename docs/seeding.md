##Seeding/Forcing
The seeding protocol to be used is specified in [config/config.yml](https://github.com/JockLawrie/Covid.jl/blob/master/config/config.yml) under `forcing:`.

Seeding is specified heirarchically through

   - Date
   - Locations to seed on the given date(given as SA2 codes or "Any")
   - Disease state to 'force' agents into within each location
   - Number of agents to seed into the given state

An illustrative example is the following:

```yaml
forcing:
	2020-06-06: {Any: {E: 10},
				 213011328: {E: 20}}
	2020-06-08: {Any: {E: 10, R:5}}
```

On 2020-06-06 10 individuals from any SA2 will become exposed and 20 individuals from SA2 213011328 will also become exposed.
On 2020-06-08 10 individuals from any SA2 will become exposed and 5 individuals will recover.