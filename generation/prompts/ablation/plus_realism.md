You are the question writer in a pipeline that creates clinical retrieval-training
data. Write ONE clinical question and 2 ideal-answer sketches, grounded ONLY in the
papers below.

## Task
QUESTION TYPE: {question_type}
ASK: the question's final sentence must be a natural phrasing of: "{ask_template}"
ENTITY: {entity}
<!-- ablation: the parent says SECRET ENTITY; with the masking rule gone the concept is no longer secret -->
SYNONYMS: {synonyms}

## Source papers
{grounding_block}

## Rules for the question
<!-- ablation: the parent fills its rule 1 slot per cell, LATENT ENTITY when the concept is hidden and NAMED ENTITY when it cannot be. This arm hard-wires the NAMED ENTITY text (prompts/rule1_named.md, verbatim) for every cell, including the question types whose tag is required. -->
1. NAMED ENTITY. This question type names its entity: you may (and normally should) name the entity in the query. There is no secrecy requirement for the entity itself.
2. GROUNDED. Every clinical fact must be traceable to the source papers. Narrative
   detail (exact age, timeline, social context) may be invented; clinical facts may
   not.
3. ONE ANSWER. A competent physician reading the question must arrive at exactly one
   defensible answer to the ask. Identify the 2-3 nearest rival answers and include
   one detail that rules each out; record them in working_notes, never in the
   question. Do NOT include any answer, recommendation, or answer-shaped hint. Do not
   restate the chosen papers' findings.
4. LENGTH: {length_instruction}
5. REGISTER: {style_instruction}

## Rules for the sketches
Write 2 invented title + abstract sketches of papers that would perfectly answer this
question, each from a different angle. These may name the entity freely. 2-4 sentences
each, realistic academic style.

## Output strict JSON
{"query": "...",
 "working_notes": {"siblings_closed": ["<rival>: <closing detail>", ...],
                   "grounding_map": {"<question element>": "<G# it comes from>", ...},
                   "banned_terms": ["<every eponym, gene, sign/score/test name you avoided naming under rule 1>", ...]},
 "hyde": [{"title": "...", "abstract": "..."}, ...]}
