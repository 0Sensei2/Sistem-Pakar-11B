from flask import Flask, request, jsonify, render_template, session
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Initialize Vectorstore and LLM
vectorstore = Chroma(
    persist_directory="hasil_data_latih",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

# Create ChatPromptTemplate with MessagesPlaceholder for conversation history
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
                "Anda adalah asisten kesehatan kelinci yang sangat berpengetahuan tentang deteksi penyakit kelinci. tugas anda memberikan informasi tentang nama penyakit yang dialami kelinci serta solusi untuk penanganannya."
                "Anda hanya akan memproses pertanyaan berdasarkan informasi yang terdapat dalam dataset."
                "Fokus jawaban Anda adalah membantu mendeteksi gejala penyakit menular pada kelinci dan memberikan informasi tentang nama penyakit yang dialami kelinci serta langkah pencegahan yang tepat tanpa menyarankan untuk pergi ke dokter."
                "Pastikan jawaban Anda tetap ringkas, relevan, bermanfaat, dan mudah dipahami oleh pemilik kelinci untuk mendukung kesehatan hewan peliharaan mereka tanpa memberikan saran yang tidak perlu."
                "Data dan informasi yang Anda berikan diperoleh melalui Dokter hewan."
                "Jika pengguna menyebutkan nama mereka, ingatlah nama tersebut dan gunakan di respons berikutnya, tetapi Anda tidak perlu menyebutkan nama mereka di setiap respons."
        ),
        # The variable_name here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create LLM chain with memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/kelinci')
def hanako():
    return render_template('kelinci.html')


# Route for handling chatbot queries
@app.route('/ask', methods=['POST'])
def ask():
    # Get query from JSON payload
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Process query using LLMChain
    response = conversation_chain.run(question=query)
    answer = response if response else "Maaf, saya tidak dapat menemukan jawaban."

 # Replace Markdown ** with <strong> for bold in HTML
    formatted_answer = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", answer)

    # Convert response into bullet points if it contains list-like structure
    formatted_answer = format_as_points(formatted_answer)

    return jsonify({"query": query, "answer": formatted_answer})


def format_as_points(answer):
    """
    Convert text containing bullet points into an HTML unordered/ordered list and remove symbols like '*'.
    """
    # Split response into lines
    lines = answer.splitlines()
    list_items = []
    for line in lines:
        # Match lines starting with numbers, asterisks, or dashes and clean up
        match = re.match(r"^\s*(\d+\.\s|[-*]\s)(.*)", line)
        if match:
            # Extract the main text and wrap it in a list item
            content = match.group(2).strip()  # Extract the text after the symbol
            list_items.append(f"<li>{content}</li>")
        else:
            # Append non-list items as regular text
            if list_items:  # Close list if no longer in list-like structure
                list_items.append("</ul>")
                list_items.append(line.strip())
                list_items.append("<ul>")
            else:
                list_items.append(line.strip())

    # Wrap detected items in an unordered list <ul>
    if list_items:
        list_items.insert(0, "<ul>")
        list_items.append("</ul>")
        return "\n".join(list_items)

    return answer

# Route to clear conversation history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    memory.clear()  # Clear the history from memory
    return jsonify({"message": "Conversation history cleared."})

if __name__ == '__main__':
    app.run(debug=True)
