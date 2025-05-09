from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os
import sqlite3
import requests
from functools import lru_cache

app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = os.urandom(24)

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# 🔁 Path to your CSV file
csv_path = r"C:\Users\ual-laptop\Downloads\augmented_faqs.csv"

# Globals
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base = []
faiss_index = None
id_to_answer = {}

# 🌼 Friendly prompt
PERSONA_PROMPT_TEMPLATE = """
You are Sunny, a kind and cheerful chatbot who always responds in a sweet and encouraging tone.
Be helpful, brief (under 400 characters), and supportive like a caring friend.
Use gentle emojis when appropriate (like 😊, 🌼, ☀️) and make the user feel welcome and appreciated.

A user asked: "{user_message}"
Here is some helpful information you know: "{matched_answer}"

Now, write a friendly and sweet reply as Sunny.
"""

# Predefined greetings
GREETINGS = {
    "hi": "Hi there! 😊 I’m so glad you stopped by. How can I help you today?",
    "hello": "Hello, sunshine! 🌼 What would you like to chat about?",
    "hey": "Hey hey! ☀️ What’s on your mind today?",
}

def load_knowledge_base():
    global knowledge_base, faiss_index, id_to_answer
    if not os.path.exists(csv_path):
        print("CSV file not found at:", csv_path)
        return

    df = pd.read_csv(csv_path)

    if 'answer_x' in df.columns and 'answer_y' in df.columns:
        df['answer'] = df[['answer_x', 'answer_y']].bfill(axis=1).iloc[:, 0]
    elif 'answer' in df.columns:
        df['answer'] = df['answer']
    else:
        print("No valid answer columns found.")
        return

    df = df[['question', 'answer']].dropna()
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    knowledge_base = df.to_dict('records')

    embeddings = model.encode(df['question'].tolist(), show_progress_bar=True)
    embs = np.array(embeddings).astype('float32')

    if embs.shape[0] == 0:
        print("No valid embeddings found.")
        return

    faiss_index = faiss.IndexFlatL2(embs.shape[1])
    faiss_index.add(embs)

    id_to_answer.clear()
    for i, row in enumerate(knowledge_base):
        id_to_answer[i] = row['answer']

    print(f"✅ Loaded {len(knowledge_base)} entries into the knowledge base.")

@lru_cache(maxsize=128)
def embed_query(text):
    return model.encode([text]).astype('float32')

def get_best_answer(user_query):
    global faiss_index

    # Check for greetings
    normalized = user_query.lower().strip()
    if normalized in GREETINGS:
        return GREETINGS[normalized], True  # True = direct answer, skip LLM

    if faiss_index is None:
        return "Knowledge base not loaded.", True

    query_embedding = embed_query(user_query)
    D, I = faiss_index.search(query_embedding, k=1)
    best_distance = D[0][0]
    best_index = I[0][0]

    # 🚫 Too far = not confident
    SIMILARITY_THRESHOLD = 0.85
    if best_distance > SIMILARITY_THRESHOLD:
        return "Sorry, 🥺 I couldn’t find a good answer for that. Can you try asking it a different way?", True

    return id_to_answer.get(best_index, "Sorry, 🌸 no answer found."), False

def initialize_model():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_input TEXT,
                  bot_response TEXT)''')
    conn.commit()
    conn.close()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'response': "Oopsie! 🌸 Could you try asking that again?"})

    matched_answer, is_direct = get_best_answer(user_message)

    # If it's a fallback or greeting, don't pass to LLM
    if is_direct:
        generated_response = matched_answer
    else:
        prompt = PERSONA_PROMPT_TEMPLATE.format(
            user_message=user_message,
            matched_answer=matched_answer
        )

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': 'mistral', 'prompt': prompt, 'stream': False},
                timeout=10
            )
            if response.ok:
                data = response.json()
                generated_response = data.get('response', matched_answer).strip()
                if len(generated_response) > 400:
                    generated_response = generated_response[:397] + "..."
            else:
                generated_response = matched_answer
        except Exception as e:
            print("LLM error:", e)
            generated_response = matched_answer

    # 💾 Save to chat log
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("INSERT INTO chatbot_log (user_input, bot_response) VALUES (?, ?)",
              (user_message, generated_response))
    conn.commit()
    conn.close()

    return jsonify({'response': generated_response})

if __name__ == '__main__':
    load_knowledge_base()
    initialize_model()
    app.run(debug=True)
