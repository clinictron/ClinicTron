# diagnosis — Ely-derived question type (233 of Ely's 1,396 questions)

```yaml
id: diagnosis
asks:
  - "what is the most likely diagnosis?"
  - "what is the most likely cause?"
tracks: [diseases, organisms, toxins]
masking: required
grounding: "{entity} clinical features presentation review"
dest:
  - "{entity} diagnosis differential"
  - "{entity} diagnostic criteria workup"
inject:
  - "{entity} diagnosis"
  - "{entity} differential diagnosis"
facet: "diagnosis"
```
