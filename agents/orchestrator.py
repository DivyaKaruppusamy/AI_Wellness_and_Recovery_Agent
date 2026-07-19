import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tools.wearable_analytics import calculate_recovery_metrics

# ==========================================
# CENTRALIZED SYSTEM STATE
# ==========================================
class AgentState:
    def __init__(self, user_query: str):
        self.user_query = user_query
        self.is_safe = True
        self.safety_escalation_message = ""
        self.analytics_data = {}
        self.rag_context = ""
        self.final_recommendation = ""
        self.agent_history = []

    def log_agent_action(self, agent_name: str, status: str):
        self.agent_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": agent_name,
            "status": status
        })


# ==========================================
# AGENT DEFINITIONS
# ==========================================

class GatekeeperAgent:
    """Agent 1: Screens inputs for critical physical warning signals."""
    def __init__(self):
        self.name = "Gatekeeper Agent"
        self.critical_triggers = ["chest pain", "chest tightness", "dizziness", "fainting", "breathlessness"]

    def execute(self, state: AgentState) -> AgentState:
        if any(trigger in state.user_query.lower() for trigger in self.critical_triggers):
            state.is_safe = False
            state.safety_escalation_message = (
                "🚨 CRITICAL SAFETY ESCALATION: "
                "I cannot provide exercise advice or analyze data when cardiopulmonary or circulatory warning symptoms "
                "(chest tightness, pain, dizziness) are mentioned. Please seek immediate professional medical attention."
            )
            state.log_agent_action(self.name, "BLOCKED: Safety Triggers Met")
        else:
            state.log_agent_action(self.name, "PASSED: Query Cleared")
        return state


class DataAnalystAgent:
    """Agent 2: Handles deterministic data computations over metrics."""
    def __init__(self, csv_path: str = "D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\data\\raw\\wearable_metrics.csv"):
        self.name = "Data Analyst Agent"
        self.csv_path = csv_path

    def execute(self, state: AgentState) -> AgentState:
        if not state.is_safe:
            state.log_agent_action(self.name, "SKIPPED")
            return state
            
        # Call your analytics utility tool
        metrics = calculate_recovery_metrics(self.csv_path)
        state.analytics_data = metrics
        state.log_agent_action(self.name, f"SUCCESS: Calculated Recovery Score ({metrics['score']})")
        return state


class KnowledgeRetrievalAgent:
    """Agent 3: Handles semantic lookups into the vector knowledge base."""
    def __init__(self):
        self.name = "Knowledge Retrieval Agent"
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def execute(self, state: AgentState) -> AgentState:
        if not state.is_safe:
            state.log_agent_action(self.name, "SKIPPED")
            return state

        try:
            db_connection = Chroma(
                collection_name="my_wellness_collection",
                persist_directory="D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db",
                embedding_function=self.embedding_model
            )
            # Use data analytics state insights to guide retrieval target focus
            recovery_tier = state.analytics_data.get("level", "Low Recovery")
            search_query = f"guidance for {recovery_tier} adaptations and active recovery"
            
            matched_chunks = db_connection.similarity_search(search_query, k=1)
            state.rag_context = matched_chunks[0].page_content if matched_chunks else "Prioritize rest."
            state.log_agent_action(self.name, "SUCCESS: Vector Context Extracted")
        except Exception as e:
            state.rag_context = "Fallback Guide: Adapt intensity downward."
            state.log_agent_action(self.name, f"ERROR: Vector DB Connection Exception: {e}")
            
        return state


class WellnessPlannerAgent:
    """Agent 4: Combines mathematical data + RAG text to formulate the final strategy."""
    def __init__(self):
        self.name = "Wellness Planner Agent"

    def execute(self, state: AgentState) -> AgentState:
        if not state.is_safe:
            state.final_recommendation = state.safety_escalation_message
            state.log_agent_action(self.name, "SKIPPED: Emitted Safety Block")
            return state

        # Synthesis Layer
        score = state.analytics_data['score']
        level = state.analytics_data['level']
        reasons = ", ".join(state.analytics_data['reasons']) if state.analytics_data['reasons'] else "No baseline anomalies."
        
        if score < 60:
            plan = "Bypass high-intensity exercise entirely. Focus strictly on active recovery paths (stretching, mobility drills, or a 25-minute walk)."
        elif score < 80:
            plan = "Proceed with scheduled split workouts but dynamically scale back total structural volume and working weights by 25%."
        else:
            plan = "Biomarkers normal. Complete standard intended training schedule while executing a deliberate dynamic warm-up."

        state.final_recommendation = (
            f"### [Orchestration Plan] Readiness Tier: {level} ({score}/100)\n"
            f"**Data Anomalies Detected:** {reasons}\n"
            f"**Retrieved Structural Guidelines:** {state.rag_context[:120]}...\n\n"
            f"**Strategic Workout Action:** {plan}"
        )
        state.log_agent_action(self.name, "SUCCESS: Final Plan Assembled")
        return state


# ==========================================
# CENTRALIZED MULTI-AGENT ORCHESTRATOR ENGINE
# ==========================================
class WellnessAgentOrchestrator:
    def __init__(self):
        self.gatekeeper = GatekeeperAgent()
        self.analyst = DataAnalystAgent()
        self.retrieval = KnowledgeRetrievalAgent()
        self.planner = WellnessPlannerAgent()

    def process(self, query: str) -> AgentState:
        # Initialize Shared Thread State
        state = AgentState(user_query=query)
        
        # Pipeline Sequential Orchestration Chain
        state = self.gatekeeper.execute(state)
        state = self.analyst.execute(state)
        state = self.retrieval.execute(state)
        state = self.planner.execute(state)
        
        return state


# Validation Verification Check
if __name__ == "__main__":
    orchestrator = WellnessAgentOrchestrator()
    
    # Test Run safe execution path
    result_state = orchestrator.process("Should I lift heavy weights today?")
    print(result_state.final_recommendation)
    print("\nAgent Trace Execution Trail:", result_state.agent_history)
