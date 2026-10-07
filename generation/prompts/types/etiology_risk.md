# etiology_risk — question type (46 of Ely's 1,396 questions)

```yaml
id: etiology_risk
asks:
  - "what causes or raises the risk of this condition?"
  - "why does this exposure produce this condition?"
tracks: [diseases, drugs, organisms, toxins]
masking: optional
grounding: "{entity} etiology risk factors pathogenesis review"
dest:
  - "{entity} risk factors etiology"
  - "{entity} pathogenesis mechanism"
inject:
  - "{entity} risk factors"
  - "{entity} etiology"
facet: "etiology and risk"
```
