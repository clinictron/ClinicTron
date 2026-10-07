You write ONE realistic clinical retrieval query for training a biomedical search model. Real physicians ask decision-embedded questions — study the examples below and write in their spirit, never their topics.

## Real physician questions of this kind (style examples — do not copy topics)
{exemplar_block}

## Seed entity
{seed_entity}

## Your task
{group_block}

## Rules
1. The question must be answerable by biomedical research papers (not drug handbooks, not normal-value tables, not local/legal knowledge).
2. Use your own medical knowledge to choose partners (drugs, comorbidities, populations) that GENUINELY relate to the seed entity.
3. Do NOT include any answer, recommendation, or answer-shaped hint. Naming the entities the question is ABOUT is allowed and often required by the task; what must never appear is any statement or clue about which answer or resolution is correct.
4. LENGTH: {length_instruction}
5. REGISTER: {style_instruction}
## Output strict JSON
{"query": "..."}
