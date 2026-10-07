# test_selection — Ely-derived question type (163 of Ely's 1,396 questions)

```yaml
id: test_selection
asks:
  - "what test should be ordered next?"
  - "what test confirms the suspicion?"
tracks: [diseases, organisms]
masking: required
grounding: "{entity} clinical features presentation review"
dest:
  - "{entity} diagnostic testing workup"
  - "{entity} confirmatory test"
inject:
  - "{entity} diagnostic test"
  - "{entity} workup imaging laboratory"
facet: "diagnostic testing"
```
