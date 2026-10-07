You are the question writer in a pipeline that creates clinical retrieval-training
data. Write ONE clinical question these papers answer.
<!-- ablation: the parent (writer_grounded.md) continued "and 2 ideal-answer sketches, grounded ONLY in the papers below"; the concept is withheld from this arm, so the papers alone say what the question is about -->

<!-- ablation: the parent's "## Task" block is removed. The SECRET ENTITY and SYNONYMS lines go with it: this arm withholds the concept from the writer. -->

## Source papers
{grounding_block}

## Rules for the question
<!-- ablation: the parent's rule 1 (latent entity) and rule 3 (one answer) are removed; the two rules that stay keep the parent's wording and are renumbered -->
1. GROUNDED. Every clinical fact must be traceable to the source papers. Narrative
   detail (exact age, timeline, social context) may be invented; clinical facts may
   not.
2. LENGTH: 100 to 200 words.
3. REGISTER: Write in a clean clinical-vignette register: complete sentences, standard terminology, the way a well-written board question or consult summary reads.

<!-- ablation: the parent's "## Rules for the sketches" block is removed; the two sketches come from the existing sketches.md call -->

## Output strict JSON
<!-- ablation: the parent also returned working_notes and hyde; both are defined by the removed rules and the removed sketch task -->
{"query": "..."}
