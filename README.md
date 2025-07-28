RMIT University Chatbot
This project implements a chatbot designed specifically for RMIT University, providing users with an AI-driven assistant to guide them through various university-related queries, such as course enrolment, career advice, and more. The chatbot leverages cutting-edge AI tools like LangChain, OpenAI, and FAISS to deliver efficient, accurate, and context-aware responses.

Features:
Backend (chatbot_brain.py): Utilizes LangChain to process and handle natural language queries, leveraging OpenAI's embeddings and a FAISS-based search index for retrieving relevant answers.

Frontend (chatbot_ui.py): Built with Streamlit, this provides an intuitive, user-friendly interface for interacting with the chatbot. It supports real-time chat, history tracking, and a clean design for an enhanced user experience.

Knowledge Base: The chatbot uses a custom knowledge base created from text documents stored in a "docs" folder. It processes and indexes these documents to answer questions efficiently.

Setup Instructions:
Clone the repository.

Create and activate a virtual environment:

Windows: python -m venv .venv

macOS/Linux: python3 -m venv .venv

Install the required dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Set up your .env file with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Run the backend to create the knowledge base:

bash
Copy
Edit
python chatbot_brain.py
Launch the frontend with:

bash
Copy
Edit
streamlit run chatbot_ui.py
Requirements:
langchain_community==0.3.24

langchain_openai==0.3.21

langchain==0.3.25

openai==1.84.0

faiss-cpu==1.11.0

python-dotenv==1.1.0

termcolor==3.1.0

streamlit==1.45.1

