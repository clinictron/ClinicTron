# test_interpretation — Ely-derived question type (64 of Ely's 1,396 questions)

```yaml
id: test_interpretation
asks:
  - "what explains this result?"
tracks: [diseases, drugs]
masking: required
grounding: "{entity} laboratory findings review"
dest:
  - "{entity} test result interpretation"
inject:
  - "{entity} interpretation"
facet: "test result interpretation"
```
