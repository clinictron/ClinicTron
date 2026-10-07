# screening_prevention — Ely-derived question type (41 of Ely's 1,396 questions)

```yaml
id: screening_prevention
asks:
  - "what screening/prophylaxis is indicated?"
tracks: [diseases, vaccines]
masking: optional
grounding: "{entity} clinical features review"
dest:
  - "{entity} screening guidelines"
  - "{entity} prevention prophylaxis"
inject:
  - "{entity} screening interval"
  - "{entity} prophylaxis indication"
facet: "screening / prevention"
```
