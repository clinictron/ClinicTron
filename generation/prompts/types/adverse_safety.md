# adverse_safety — Ely-derived question type (114 of Ely's 1,396 questions)

```yaml
id: adverse_safety
asks:
  - "which medication or exposure is responsible?"
  - "is this agent safe for this patient?"
tracks: [drugs, toxins, vaccines]
masking: required
grounding: "{entity} clinical use review"
dest:
  - "{entity} adverse effects safety"
  - "{entity} toxicity interactions"
inject:
  - "{entity} adverse effects"
  - "{entity} drug interaction safety"
facet: "safety"
```
