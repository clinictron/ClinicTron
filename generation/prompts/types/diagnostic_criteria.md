# diagnostic_criteria — question type (43 of Ely's 1,396 questions)

```yaml
id: diagnostic_criteria
asks:
  - "what are the diagnostic criteria for this condition?"
  - "does this presentation fit the condition?"
tracks: [diseases, organisms]
masking: impossible
grounding: "{entity} diagnostic criteria clinical manifestations review"
dest:
  - "{entity} diagnostic criteria"
  - "{entity} clinical manifestations presentation"
inject:
  - "{entity} diagnostic criteria"
  - "{entity} manifestations"
facet: "diagnostic criteria"
```
