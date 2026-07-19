import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tools.wearable_analytics import calculate_recovery_metrics
from agents.orchestrator import WellnessAgentOrchestrator

# Page configuration
st.set_page_config(page_title="AI Wellness & Recovery Agent", layout="wide", page_icon="💪")

# Initialize session states for Human Approval and Logs
if "approval_status" not in st.session_state:
    st.session_state.approval_status = "Pending Action"
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []

# # Core Agent Processing Pipeline
# def process_agent_request(user_query, csv_path="data/raw/wearable_metrics.csv"):
#     start_time = datetime.now()
    
#     # 1. Safety Guardrail Evaluation Node
#     critical_triggers = ["chest pain", "chest tightness", "dizziness", "fainting", "breathlessness", "injury"]
#     if any(trigger in user_query.lower() for trigger in critical_triggers):
#         log_entry = {
#             "timestamp": start_time.strftime("%H:%M:%S"),
#             "query": user_query,
#             "safety_flag": "TRIGGERED",
#             "status": "Blocked",
#             "latency": f"{(datetime.now() - start_time).total_seconds():.3f}s"
#         }
#         st.session_state.execution_logs.insert(0, log_entry)
#         return "SAFETY_BLOCKED", None, None

#     # 2. Structured Wearable Ingestion & Calculation Node
#     analytics = calculate_recovery_metrics(csv_path)
    
#     # 3. RAG Retrieval Node (HuggingFace vector space lookup)
#     try:
#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         db_connection = Chroma(
#             collection_name="my_text_collection",
#             persist_directory="./chroma_db",
#             embedding_function=embedding_model
#         )
#         search_intent = f"guidance for {analytics['level']} and workout modifications"
#         matched_chunks = db_connection.similarity_search(search_intent, k=1)
#         rag_context = matched_chunks[0].page_content if matched_chunks else "Listen to your body and adjust training."
#     except Exception as e:
#         rag_context = f"Fallback Guide: Maintain low-impact adjustments due to lower recovery score. (DB Link Exception: {e})"

#     latency = f"{(datetime.now() - start_time).total_seconds():.3f}s"
    
#     # Track Monitoring Data
#     log_entry = {
#         "timestamp": start_time.strftime("%H:%M:%S"),
#         "query": user_query,
#         "safety_flag": "PASSED",
#         "status": "Success",
#         "score": analytics['score'],
#         "latency": latency
#     }
#     st.session_state.execution_logs.insert(0, log_entry)
    
#     return "SUCCESS", analytics, rag_context

def process_agent_request(user_query):
    orchestrator = WellnessAgentOrchestrator()
    final_state = orchestrator.process(user_query)
    
    # Check if gatekeeper caught a critical flag
    if not final_state.is_safe:
        return "SAFETY_BLOCKED", None, None
        
    return "SUCCESS", final_state.analytics_data, final_state.rag_context


# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("🏃 AI Wellness, Recovery & Workout Readiness Dashboard")
st.markdown("---")

# Sidebar for Ingesting Raw Data Views
st.sidebar.header("📁 Wearable Analytics Engine")
try:
    df_raw = pd.read_csv("data/raw/wearable_metrics.csv")
    st.sidebar.dataframe(df_raw.tail(3), use_container_width=True)
    st.sidebar.success("Linked to historical metrics pipeline.")
except Exception:
    st.sidebar.error("Missing data/raw/wearable_metrics.csv file.")

# MAIN INTERACTION ZONE: User Input Panel
user_input = st.text_input(
    "💬 Ask your Agent a workout or readiness question:",
    placeholder="e.g., 'My resting heart rate is high and sleep was poor. Suggest today's plan.'",
)

if user_input:
    status, metrics, rag_text = process_agent_request(user_input)
    
    # CASE A: Safety Guardrail Triggered
    if status == "SAFETY_BLOCKED":
        st.error("🚨 **CRITICAL SAFETY ESCALATION TRIGGERED**")
        st.warning(
            "I cannot recommend exercise routines or evaluate training data while warning indicators like "
            "chest discomfort, tightness, or dizziness are actively reported. Please immediately stop physical activity "
            "and consult a qualified healthcare professional or visit an urgent care clinic."
        )
        
    # CASE B: Successful Agent Execution Flow
    elif status == "SUCCESS":
        # Create Layout Columns for Stats vs Recommendations
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📊 Computed Health Baselines")
            
            # Display Big Metric Blocks
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Recovery Score", f"{metrics['score']}/100", delta=metrics['level'])
            with m_col2:
                st.metric("Today's HR / Baseline", f"{metrics['today_metrics']['resting_heart_rate']} bpm", 
                          f"Avg: {metrics['baselines']['avg_hr']:.1f}")
                
            # Plot historical baseline charts using Plotly
            st.markdown("**Historical HRV & Stress Trends**")
            fig = px.line(df_raw, x="date", y=["hrv", "stress_score"], markers=True, 
                          labels={"value": "Score Units", "variable": "Metric"})
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("🤖 Generated Recommendation Block")
            
            # Output Box Displaying Core Decisions
            st.info(f"**Target System Routing Layer:** {metrics['routing']}")
            
            # Reasons Bullet List
            st.markdown("**System Anomaly Analysis:**")
            if metrics['reasons']:
                for r in metrics['reasons']:
                    st.markdown(f"❌ {r}")
            else:
                st.markdown("✅ All checked biomarker trends matching your running historical average baseline.")
                
            # RAG Context View Container
            with st.expander("📖 View Retrieved RAG Source Material Context"):
                st.write(rag_text)
                
            # Tailored Output Recommendation
            st.markdown("### 📋 Suggested Plan Adjustment")
            if metrics['score'] < 60:
                suggested_text = "Pivot directly away from high-intensity work today. Substitute with a light 30-minute outdoor walk or focal physical recovery mobility drills. Prioritise baseline hydration targets."
            elif metrics['score'] < 80:
                suggested_text = "Complete your planned workout split but reduce total volume by 20-30%. Avoid max-effort fatigue sets and elongate your baseline rest intervals."
            else:
                suggested_text = "Your physical metrics show normal adaptation vectors. Proceed with your planned target workout intensity; remember to warm up completely."
                
            st.success(suggested_text)
            
            # ==========================================
            # HUMAN APPROVAL COMPONENT LAYER
            # ==========================================
            st.markdown("### ⚖️ Human In-The-Loop Approval Node")
            st.write(f"Current Execution Change State: **{st.session_state.approval_status}**")
            
            app_col1, app_col2, app_col3 = st.columns(3)
            with app_col1:
                if st.button("👍 Approve Adjustment"):
                    st.session_state.approval_status = "Approved & Saved to Schedule"
                    st.rerun()
            with app_col2:
                if st.button("✏️ Modify Recommendation"):
                    st.session_state.approval_status = "Flagged for Manual Modification"
                    st.rerun()
            with app_col3:
                if st.button("👎 Reject & Keep Original"):
                    st.session_state.approval_status = "Rejected (Maintained Current Routine)"
                    st.rerun()

# ==========================================
# MONITORING & LOGGING METRICS VIEW
# ==========================================
st.markdown("---")
st.subheader("🕵️ System Trace & Operations Monitor Logs")
if st.session_state.execution_logs:
    st.dataframe(pd.DataFrame(st.session_state.execution_logs), use_container_width=True)
else:
    st.info("No query trace logs available yet. Submit a question above to see backend trace metrics live.")

# General Non-clinical Disclaimer Required for Health adjacent frameworks
st.markdown("---")
st.caption(
    "⚠️ **General Educational Disclaimer:** This system provides basic health tracker metric logging and educational wellness information only. It does not output personalized clinical insights or medical diagnostics. Fitness tracking data should not replace explicit physical or painful somatic feedback. For prolonged exhaustion or acute symptoms, immediately seek out an evaluation from a primary medical doctor or clinical emergency service."
)
