"""
Quick accuracy benchmark for the Hallucination Firewall.
Uses 10 patients from the indexed records with known ground truth.
Measures: correct corrections, false positives, false negatives.
"""
import pandas as pd
from data_processor import normalize_name, DATASET_PATH
from firewall import chat

ORG_ID = "hospital"

def get_test_cases(n=10):
    df = pd.read_csv(DATASET_PATH)
    # Use the same 500 indexed records (random_state=42)
    indexed = df.sample(n=500, random_state=42).reset_index(drop=True)
    # Pick n patients for testing
    sample = indexed.sample(n=n, random_state=99).reset_index(drop=True)
    return [
        {
            "question":          f"What medication is {normalize_name(row['Name'])} currently on?",
            "correct_answer":    row["Medication"],
            "correct_doctor":    row["Doctor"],
            "correct_condition": row["Medical Condition"],
        }
        for _, row in sample.iterrows()
    ]

def run_benchmark():
    cases = get_test_cases(10)
    print(f"Running benchmark on {len(cases)} patients...\n")
    print(f"{'#':<3} {'Patient Question':<50} {'Status':<12} {'Correct Med':<15} {'Answer Contains?'}")
    print("-" * 105)

    correct = 0
    corrected_right = 0
    corrected_wrong  = 0
    verified_right   = 0
    verified_wrong   = 0

    for i, case in enumerate(cases, 1):
        result = chat(case["question"], ORG_ID)
        status = result["status"]
        final  = result["final_answer"].lower()
        med    = case["correct_answer"].lower()
        hit    = med in final

        # Count outcomes
        if status == "CORRECTED" and hit:
            corrected_right += 1
            correct += 1
        elif status == "CORRECTED" and not hit:
            corrected_wrong += 1
        elif status == "VERIFIED" and hit:
            verified_right += 1
            correct += 1
        elif status == "VERIFIED" and not hit:
            verified_wrong += 1

        mark = "✅" if hit else "❌"
        print(f"{i:<3} {case['question'][:48]:<50} {status:<12} {case['correct_answer']:<15} {mark}")

    total = len(cases)
    print(f"\n{'='*60}")
    print(f"RESULTS ({total} test cases)")
    print(f"{'='*60}")
    print(f"Overall accuracy:          {correct}/{total} ({100*correct//total}%)")
    print(f"Caught + corrected right:  {corrected_right}")
    print(f"Corrected but still wrong: {corrected_wrong}  (false correction)")
    print(f"Verified + was correct:    {verified_right}")
    print(f"Missed hallucination:      {verified_wrong}  (false negative)")

if __name__ == "__main__":
    run_benchmark()
