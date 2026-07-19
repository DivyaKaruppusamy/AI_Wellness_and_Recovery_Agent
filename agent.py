import sys
import os

# Ensure local modules can be loaded smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import WellnessAgentOrchestrator

def main():
    print("=" * 60)
    print("🏃 AI WELLNESS MULTI-AGENT HARNESS RUNNER")
    print("=" * 60)
    
    orchestrator = WellnessAgentOrchestrator()
    
    # Scenario 1: Standard Workout Intent (Triggers Low Recovery Pipeline)
    query_1 = "Should I run a high-intensity HIIT training block today?"
    print(f"\n[Executing Query 1]: '{query_1}'")
    state_1 = orchestrator.process(query_1)
    print(state_1.final_recommendation)
    print("\nExecution Trail Traces:")
    for trace in state_1.agent_history:
        print(f" - [{trace['timestamp']}] {trace['agent']}: {trace['status']}")
        
    print("\n" + "="*60 + "\n")
    
    # Scenario 2: Safety Boundary Intent (Triggers Gatekeeper Intercept Node)
    query_2 = "I want to train legs but I have sharp chest tightness and dizziness."
    print(f"[Executing Query 2]: '{query_2}'")
    state_2 = orchestrator.process(query_2)
    print(state_2.final_recommendation)
    print("\nExecution Trail Traces:")
    for trace in state_2.agent_history:
        print(f" - [{trace['timestamp']}] {trace['agent']}: {trace['status']}")

if __name__ == "__main__":
    main()
