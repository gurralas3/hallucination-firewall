import pandas as pd
from data_processor import normalize_name, DATASET_PATH
from firewall import chat

def get_hospital_questions():
    df = pd.read_csv(DATASET_PATH)
    patients = [normalize_name(r["Name"]) for _, r in df.sample(n=5, random_state=42).iterrows()]
    return [
        (f"What medication is {patients[0]} currently on?",    "hospital"),
        (f"What is {patients[1]}'s medical condition?",        "hospital"),
        (f"Who is the doctor treating {patients[2]}?",         "hospital"),
        (f"What were the test results for {patients[3]}?",     "hospital"),
    ]

LEGAL_QUESTIONS = [
    ("Who was the Chief Justice in DRAPER v. UNITED STATES?",                        "legal"),
    ("What was the outcome of FEDERAL TRADE COMMISSION v. NATIONAL LEAD CO.?",       "legal"),
    ("How many majority votes were in ANDERSON et al. v. CELEBREZZE?",               "legal"),
    ("When was UNITED STATES v. R. F. BALL CONSTRUCTION CO. decided?",               "legal"),
]

FINANCE_QUESTIONS = [
    ("What is Paul Watts's account balance?",             "finance"),
    ("What type of account does Lisa Fuentes have?",      "finance"),
    ("What is Scott Villa's account balance?",            "finance"),
    ("When did Teresa Bell register her account?",        "finance"),
]

if __name__ == "__main__":
    all_questions = get_hospital_questions() + LEGAL_QUESTIONS + FINANCE_QUESTIONS
    print(f"Pre-warming cache for {len(all_questions)} questions across 3 domains...\n")

    for i, (question, org_id) in enumerate(all_questions, 1):
        print(f"[{i}/{len(all_questions)}] [{org_id.upper()}] {question}")
        result = chat(question, org_id)
        print(f"  Status: {result['status']}")
        print(f"  Answer: {result['final_answer'][:90]}...")
        print()

    print("Cache warmed. All 3 domains ready for instant demo responses.")
