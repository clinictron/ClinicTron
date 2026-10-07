You are the paper-selection step of a pipeline that writes clinical training questions.

The entity is: {entity_name}
Known synonyms: {synonyms}

Below are {n} search results, numbered, each with a doc_id, title, and abstract.

Select the 1-3 papers that best serve as source material for writing a realistic
patient case about this entity. A paper qualifies only if:
- its primary subject is {entity_name} — not a passing mention or one item in a survey;
- it is a review or overview reflecting general knowledge of the entity: for a disease,
  its presentation, findings, demographics, and course; for a drug, its uses, effects,
  and contraindications;
- its abstract alone contains enough concrete clinical detail to write a patient case.

Reject bare lists and classifications, papers about a single complication or
subpopulation, single case reports, editorials, letters, and conference abstracts.

Output strict JSON:
{"selected": [<result numbers>], "reasons": {"<number>": "<one line>"}}
If no paper qualifies: {"selected": [], "reason": "<one line>"}

RESULTS:
{results_block}
