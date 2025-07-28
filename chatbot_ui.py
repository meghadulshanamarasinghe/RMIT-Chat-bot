import streamlit as st
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()

# Custom CSS to style the chatbot interface
st.markdown(
    """
    <style>
    /* Main background of the application */
    .stApp {
        background-color: white !important;
    }
    
    /* Container for messages to control alignment */
    .message-container {
        width: 100%;
        display: flex;
        margin-bottom: 8px;
    }
    
    .user-message-container {
        justify-content: flex-end;
    }
    
    .bot-message-container {
        justify-content: flex-start;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 8px 12px;
        border-radius: 10px 10px 0 10px;
        max-width: 70%;
        word-wrap: break-word;
        border: none !important;  /* Remove border */
    }
    
    .bot-message {
        background-color: #e9ecef;
        color: black;
        padding: 8px 12px;
        border-radius: 10px 10px 10px 0;
        max-width: 70%;
        word-wrap: break-word;
        border: none !important;  /* Remove border */
    }
    
    .message-timestamp {
        font-size: 0.7em;
        color: #666;
        margin-top: 4px;
    }
    
    .user-timestamp {
        text-align: right;
    }
    
    .bot-timestamp {
        text-align: left;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        border: none !important;  /* Remove border */
        border-radius: 5px !important;
        padding: 8px 12px !important;
        box-shadow: 0 0 0 1px #ced4da !important;  /* Subtle shadow instead of border */
    }
    
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;  /* Remove border */
        border-radius: 5px !important;
        padding: 8px 16px !important;
        height: 38px !important;
        margin-top: 1px !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        background-color: #0056b3 !important;
    }
    
    /* Flex container for input + button */
    .input-row {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 20px;
    }
    
    .header-title {
        color: black !important;
        font-size: 1.8em;
        margin-bottom: 0;
    }
    
    .header-subtitle {
        color: black !important;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state to store database and chat history
if "db" not in st.session_state:
    try:
        embeddings = OpenAIEmbeddings()
        st.session_state.db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.session_state.chat_history = []
        st.session_state.last_query = ""
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        st.session_state.db = None
        st.session_state.chat_history = []
        st.session_state.last_query = ""

# Display the header section
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">Welcome to the RMIT Course Enrolment Chatbot</h1>
        <p class="header-subtitle">Your personal guide to course enrolment, career advice, and more at RMIT University.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create and initialize the chat container
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

# Add welcome message if chat history is empty
if not st.session_state.chat_history:
    st.session_state.chat_history.append({
        "role": "bot",
        "content": "Hello! How can I assist you with your RMIT course enrolment or other university matters today?",
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

# Display the chat history with proper alignment
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="message-container user-message-container">
                <div>
                    <div class="user-message">{message["content"]}</div>
                    <div class="message-timestamp user-timestamp">{message["timestamp"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="message-container bot-message-container">
                <div>
                    <div class="bot-message">{message["content"]}</div>
                    <div class="message-timestamp bot-timestamp">{message["timestamp"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# Create a form for user input
with st.form(key="chat_form"):
    cols = st.columns([0.85, 0.15])
    with cols[0]:
        user_input = st.text_input(
            "Your question:", 
            placeholder="e.g., How do I enrol in COSC1111?", 
            key="input",
            label_visibility="collapsed"
        )
    with cols[1]:
        submit_button = st.form_submit_button("Send", use_container_width=True)

# Handle form submission and process the query
if submit_button:
    if user_input and user_input.strip():
        if user_input.strip() != st.session_state.get("last_query", ""):
            if st.session_state.db:
                try:
                    timestamp = datetime.now().strftime("%I:%M %p")
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input.strip(),
                        "timestamp": timestamp
                    })
                    
                    st.session_state.last_query = user_input.strip()

                    llm = OpenAI()
                    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})
                    chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=False
                    )
                    response = chain.invoke({"query": user_input.strip()})["result"]

                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": response,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": f"Error: {e}",
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
            else:
                st.error("Knowledge base not loaded. Please ensure the backend has been run or check the docs folder.")
        st.rerun()
    else:
        st.warning("Please enter a question.")