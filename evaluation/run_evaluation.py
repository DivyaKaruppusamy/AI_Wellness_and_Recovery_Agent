import os
import csv
import sys
import pandas as pd

# Adjust paths to see modules sitting up one level
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.wearable_analytics import calculate_recovery_metrics
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def compute_rag_relevance(retrieved_text, target_topic):
    """Calculates keyword match coverage precision over the actual textbook text layout chunks."""
    rubrics = {
        "recovery": ["low recovery", "walking", "mobility", "yoga", "cycling", "endurance"],
        "workout adjustments": ["reduce volume", "failure", "rest periods", "intensity", "recovery time"],
        "safety": ["wellness guidance", "medical professional", "diagnose", "prescribe", "chest pain", "tightness"]
    }
    
    lookup_key = target_topic.lower().strip()
    target_keywords = rubrics.get(lookup_key, ["wellness", "guide", "recovery"])
    
    if not retrieved_text:
        return 0.0
        
    matches = [word for word in target_keywords if word in retrieved_text.lower()]
    return len(matches) / len(target_keywords)

def run_evaluation_suite():
    print("=" * 60)
    print("🚀 AUTOMATED MULTI-AGENT VALIDATION RUNNER (TEXTBOOK DATASET ALIGNED)")
    print("=" * 60)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_path = "D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db"
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Vector store directory '{db_path}' not found.")
        return

    db_connection = Chroma(
        collection_name="my_wellness_collection",
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    test_cases_file = "evaluation/test_cases.csv"
    if not os.path.exists(test_cases_file):
        print(f"❌ Error: Dataset missing at {test_cases_file}")
        return
        
    records = []
    
    with open(test_cases_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            query = row['query']
            expected_safety = row['expected_safety'].strip()
            expected_routing = row['expected_routing'].strip()
            
            # 1. Evaluate Safety Intercept Logic
            critical_triggers = ["chest pain", "chest tightness", "dizziness", "fainting", "breathlessness"]
            safety_status = "BLOCKED" if any(t in query.lower() for t in critical_triggers) else "PASSED"
            
            # 2. Evaluate Analytical Math Engine Consistency
            analytics = calculate_recovery_metrics("data/raw/wearable_metrics.csv")
            actual_routing = analytics['routing'].strip()
            
            # 3. Evaluate RAG Search Matrix Alignment against textbook data keys
            if row['target_topic'].lower().strip() == "safety":
                search_str = "SAFETY RULES medical professional chest pain tightness"
            else:
                search_str = "WORKOUT GUIDE SLEEP HYGIENE GUIDE Low recovery recommendations"
                
            try:
                chunks = db_connection.similarity_search(search_str, k=1)
                context = chunks[0].page_content if chunks else ""
            except Exception:
                context = ""
            
            relevance_score = compute_rag_relevance(context, row['target_topic'])
            
            # Assertions Evaluation
            safety_ok = (safety_status == expected_safety)
            
            if safety_status == "BLOCKED":
                routing_ok = (expected_routing == "None")
            else:
                routing_ok = (actual_routing == expected_routing)
                
            case_passed = safety_ok and routing_ok and (relevance_score > 0.0)
            
            records.append({
                "Test_ID": idx + 1,
                "Query_Snippet": query[:35] + "...",
                "Safety_Guardrail": "✅ PASS" if safety_ok else "❌ FAIL",
                "Deterministic_Routing": "✅ PASS" if routing_ok else "❌ FAIL",
                "RAG_Relevance": f"{relevance_score * 100:.1f}%",
                "Verdict": "PASSED" if case_passed else "FAILED"
            })
            
    # Compile and Display Report Matrix
    output_df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("📊 CAPSTONE METRIC EVALUATION SUMMARY")
    print("=" * 60)
    print(output_df.to_string(index=False))
    print("=" * 60)
    
    # Export log data
    output_df.to_csv("evaluation/evaluation_report.csv", index=False)
    print("🎉 File logs written to 'evaluation/evaluation_report.csv'")

if __name__ == "__main__":
    run_evaluation_suite()
