"""R2MED macro-8 and the three task-group means (nDCG@10 x100) from per-task scores.

Input: a JSON file mapping task name to nDCG@10, e.g. {"r2med_biology": 0.488, ...}.
Groups follow R2MED (Li et al. 2025). Means use unrounded per-task values."""
import json, sys

GROUPS = {
    "Q&A Reference": ["r2med_biology", "r2med_bioinformatics", "r2med_medical_sciences"],
    "Clinical Evidence": ["r2med_medxpert_exam", "r2med_medqa_diag"],
    "Clinical Case": ["r2med_pmc_treatment", "r2med_pmc_clinical", "r2med_iiyi_clinical"],
}


def mean(xs):
    return 100 * sum(xs) / len(xs)


if __name__ == "__main__":
    s = json.load(open(sys.argv[1]))
    out = {"Macro-8": mean([s[t] for ts in GROUPS.values() for t in ts])}
    out.update({g: mean([s[t] for t in ts]) for g, ts in GROUPS.items()})
    print(json.dumps({k: round(v, 2) for k, v in out.items()}))
