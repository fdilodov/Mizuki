#!/usr/bin/env python
import streamlit as st
import pandas as pd
import os
import re
# Import your existing setup logic
from chatbot import setup_embeddings, setup_llm, format_docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Mizu (水) - Hyper-Kamiokande Assistant", 
    page_icon="./images/mizu.png",
    layout="wide"  
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
        /* Hide Streamlit Default UI elements */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stHeader"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. PERSONA DEFINITIONS ---
PERSONAS = {
    "Mizu Doc": {
        "icon": "./images/mizu_doc.png",
        "role": "Documentation Specialist",
        "description": "Expert in Technical Notes, Reports, and Organization.",
        "system_prompt": "You are Mizu Doc, a documentation specialist for Hyper-Kamiokande. Focus on summarizing technical reports, locating specific TDR sections, and explaining the organizational structure."
    },
    "Mizu Tech": {
        "icon": "./images/mizu_tech.png",
        "role": "Detector Engineer",
        "description": "Expert in Hardware, PMTs, and Excavation.",
        "system_prompt": "You are Mizu Tech, a hardware engineer for Hyper-Kamiokande. Focus on the details of PMT modules, water systems, tank construction, and excavation processes. Be technical and precise about dimensions and materials."
    },
    "Mizu Soft": {
        "icon": "./images/mizu_soft.png",
        "role": "Computing Expert",
        "description": "Expert in WCSim, fiTQun, and Data Processing.",
        "system_prompt": "You are Mizu Soft, a software expert for Hyper-Kamiokande. Focus on simulation software (WCSim), reconstruction algorithms (fiTQun), and computing infrastructure. Provide Python or C++ code snippets where relevant."
    },
    "Mizu Phys": {
        "icon": "./images/mizu_phys.png",
        "role": "Physics Analyst",
        "description": "Expert in Neutrinos, Parameters, and Sensitivity.",
        "system_prompt": "You are Mizu Phys, a theoretical physicist for Hyper-Kamiokande. Focus on neutrino oscillation parameters, CP violation sensitivity, proton decay, and physics goals. Use LaTeX for all formulas."
    }
}

# --- 4. PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔒 Mizu Access Restricted")
        st.info("Please enter the access code to proceed.")
        password_input = st.text_input("Access Code", type="password", label_visibility="collapsed")

        if password_input:
            if password_input == st.secrets["APP_PASSWORD"]:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("😕 Incorrect code. Access denied.")
    return False

if not check_password():
    st.stop()

# =========================================================
#  APP LOGIC STARTS HERE
# =========================================================

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = "Mizu Doc" # Default

# Layout: Logo + Title
col1, col2 = st.columns([0.05, 0.95], vertical_alignment="center")
with col1:
    st.image("./images/mizuhkai_logo.png", width=100)
with col2:
    st.title("Mizu (水): Hyper-Kamiokande AI Hub")

# Load Resources
@st.cache_resource
def load_resources():
    vectorstore = setup_embeddings()
    llm = setup_llm()
    return vectorstore, llm

vectorstore, llm = load_resources()

# --- SIDEBAR SELECTION & STATUS ---
with st.sidebar:
    st.title("🧠 AI Persona")
    
    # 1. Dropdown to choose the brain
    # We use index=0, 1, 2 etc to match the current selection
    persona_names = list(PERSONAS.keys())
    try:
        current_index = persona_names.index(st.session_state.selected_persona)
    except ValueError:
        current_index = 0

    selected_role = st.selectbox(
        "Choose your Assistant:",
        persona_names,
        index=current_index
    )
    
    # Check if persona changed to clear chat (optional but recommended)
    if selected_role != st.session_state.selected_persona:
        st.session_state.selected_persona = selected_role
        st.session_state.messages = [] # Clear history on switch
        st.rerun()

    # Get data for current selection
    current_p = PERSONAS[selected_role]
    
    # Show the specific icon and description
    st.image(current_p["icon"], width=80)
    st.caption(current_p["description"])
    
    st.markdown("---")
    
    # 2. System Status
    st.header("System Status")
    if os.path.exists("chroma_db"):
        st.success("Mizu Memory: Online")
    else:
        st.error("Memory Not Found. Run ingest.py first!")

    st.markdown("---")
    
    # 3. Sources
    st.header("Sources")
    sources_path = "./documents/sources.txt"
    if os.path.exists(sources_path):
        with open(sources_path, "r") as f:
            st.markdown(f.read())
    else:
        st.info(f"File not found: {sources_path}")

    st.markdown("---")
    
    # 4. Clear Conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "assistant":
        # Use the icon of the CURRENTLY selected persona
        # (Or you could save the icon used at the time in the message dict)
        avatar_icon = PERSONAS[st.session_state.selected_persona]["icon"]
    else:
        avatar_icon = "./images/scientist.png" 

    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Verified Sources"):
                st.markdown(message["sources"])

# Helper Function
def find_best_sentence(text, query):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    query_words = set(query.lower().split())
    best_sentence = text[:200] + "..." 
    max_overlap = -1
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        if overlap > max_overlap and len(sentence.strip()) > 10: 
            max_overlap = overlap
            best_sentence = sentence.strip()
    return best_sentence

# Chat Input Logic
if prompt_input := st.chat_input(f"Ask {st.session_state.selected_persona} a question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    with st.chat_message("user", avatar="./images/scientist.png"):
        st.markdown(prompt_input)

    current_p_data = PERSONAS[st.session_state.selected_persona]

    with st.chat_message("assistant", avatar=current_p_data["icon"]):
        with st.spinner(f"{st.session_state.selected_persona} is analyzing..."):
            
            results = vectorstore.similarity_search_with_score(prompt_input, k=3)
            valid_docs = [doc for doc, score in results if (1.0 - (score/2)) >= 0.5]

            if not valid_docs:
                response = "I searched the database but couldn't find specific details on that topic within the current Hyper-K documentation."
                source_text = "No relevant documents found."
            else:
                context = format_docs(valid_docs)
                
                # --- DYNAMIC PROMPT INJECTION ---
                # We retrieve the specific prompt for the selected persona
                specific_persona_prompt = current_p_data["system_prompt"]
                
                sys_prompt = ChatPromptTemplate.from_template(
                    f"{specific_persona_prompt}\n"  # <--- INJECTED HERE
                    "Guidelines:\n"
                    "1. Answer strictly based on the Context below.\n"
                    "2. If writing math/physics formulas, ALWAYS use LaTeX formatting (e.g., $E=mc^2$).\n"
                    "3. If writing code, use Markdown code blocks.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}"
                )
                chain = (sys_prompt | llm | StrOutputParser())
                response = chain.invoke({"context": context, "question": prompt_input})
                
                formatted_sources = []
                for doc in valid_docs:
                    full_path = doc.metadata.get('source', 'Unknown')
                    source_name = os.path.basename(full_path)
                    page_num = doc.metadata.get('page_number', 'Unknown')
                    relevant_sentence = find_best_sentence(doc.page_content, prompt_input)
                    formatted_sources.append(f"**{source_name} (Page {page_num})**\n> \"{relevant_sentence}\"")
                
                source_text = "\n\n".join(formatted_sources)

            st.markdown(response)
            if valid_docs:
                with st.expander("View Verified Sources"):
                    st.markdown(source_text)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "sources": source_text
            })

            
