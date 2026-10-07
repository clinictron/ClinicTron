# treatment — Ely-derived question type (235 of Ely's 1,396 questions)

```yaml
id: treatment
asks:
  - "what treatment should be started?"
tracks: [diseases, organisms, toxins]
masking: required
grounding: "{entity} clinical features presentation review"
dest:
  - "{entity} treatment"
  - "{entity} management therapy options"
inject:
  - "{entity} treatment"
  - "{entity} first-line therapy"
facet: "treatment"
```
