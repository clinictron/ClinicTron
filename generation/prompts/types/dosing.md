# dosing — Ely-derived question type (134 of Ely's 1,396 questions)

```yaml
id: dosing
asks:
  - "what dose/regimen is appropriate?"
tracks: [drugs]
masking: impossible
grounding: "{entity} clinical pharmacology review"
dest:
  - "{entity} dosing"
  - "{entity} dose adjustment renal hepatic"
inject:
  - "{entity} dose reduction criteria"
  - "{entity} dosing regimen"
facet: "dosing"
```
