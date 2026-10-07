# epidemiology_prognosis — Ely-derived question type (39 of Ely's 1,396 questions)

```yaml
id: epidemiology_prognosis
asks:
  - "what is the expected course/outlook?"
tracks: [diseases]
masking: required
grounding: "{entity} clinical features presentation review"
dest:
  - "{entity} prognosis outcomes"
  - "{entity} natural history survival"
inject:
  - "{entity} prognosis"
  - "{entity} long-term outcome"
facet: "prognosis"
```
