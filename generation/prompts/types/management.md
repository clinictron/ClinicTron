# management — Ely-derived question type (75 of Ely's 1,396 questions)

```yaml
id: management
asks:
  - "admit or discharge?"
  - "how should this be managed?"
tracks: [diseases, procedures]
masking: required
grounding: "{entity} management review"
dest:
  - "{entity} severity score criteria"
  - "{entity} admission criteria management"
inject:
  - "{entity} admission criteria"
  - "{entity} severity score management threshold"
facet: "disposition / management criteria"
```
