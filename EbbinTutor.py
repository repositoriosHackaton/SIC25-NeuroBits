import json
import random
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import os
import streamlit as st
import asyncio
import requests
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit.components.v1 as components
import time
import whisper

load_dotenv()  # Cargar variables de entorno
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')
ELEVEN_LABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

whisper_model = whisper.load_model("base")  

def transcribe_audio_whisper(audio_base64):
    try:

        audio_data = base64.b64decode(audio_base64)
        
        with open("temp_audio.wav", "wb") as audio_file:
            audio_file.write(audio_data)
    
        result = whisper_model.transcribe("temp_audio.wav", language="es")
        transcribed_text = result["text"].strip()
        
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        
        if transcribed_text:
            return transcribed_text
        else:
            st.warning("No se detectó texto en el audio.")
            return "No se pudo transcribir el audio."
    except Exception as e:
        st.error(f"Error al transcribir con Whisper: {str(e)}")
        return "No se pudo transcribir el audio."

st.set_page_config(
    page_title="EbbinTutor - Tutor Académico IA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    body {
        background-color: rgba(20, 20, 20, 0.96);
        color: #E0E0E0;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }
    .stApp {
        max-width: 1700px;
        margin: auto;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .chat-message:hover {
        transform: scale(1.02);
    }
    .chat-message.user {
        background-color: rgba(58, 74, 90, 0.85);
        border-left: 5px solid #4682B4;
    }
    .chat-message.bot {
        background-color: rgba(47, 62, 70, 0.85);
        border-left: 5px solid #2E8B57;
    }
    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .chat-message .message {
        flex-grow: 1;
        color: #D3D3D3;
        font-size: 1rem;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #A9A9A9;
        margin-top: 0.5rem;
        text-align: right;
    }
    .stButton > button {
        background-color: rgba(46, 139, 87, 0.9);
        color: #FFFFFF;
        border: none;
        border-radius: 25px;
        padding: 12px 25px;
        font-size: 1rem;
        transition: background-color 0.3s ease, transform 0.2s;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: rgba(39, 122, 71, 0.9);
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 12px 15px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        background-color: rgba(51, 51, 51, 0.8);
        color: #E0E0E0;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4682B4;
        outline: none;
    }
    .topic-chip {
        display: inline-block;
        background-color: rgba(53, 79, 82, 0.8);
        border-radius: 20px;
        padding: 8px 15px;
        margin: 5px;
        font-size: 0.9rem;
        color: #B0C4DE;
        border: 1px solid rgba(82, 121, 111, 0.8);
        transition: background-color 0.3s;
    }
    .topic-chip:hover {
        background-color: rgba(82, 121, 111, 0.9);
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #A9A9A9;
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(30, 30, 30, 0.9);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
    }
    .precision {
        font-size: 0.85rem;
        color: #2E8B57;
        margin-top: 0.5rem;
        text-align: right;
    }
    .sidebar-footer {
        text-align: center;
        padding: 0.5rem;
        color: #A9A9A9;
        background-color: rgba(30, 30, 30, 0.9);
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
        position: sticky;
        bottom: 0;
        width: 100%;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Clase EbbinTutorChatbot
class EbbinTutorChatbot:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('spanish'))
        self.intents = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.model = None
        self.model_path = 'chatbot_ebbin_tutor_model.keras'
        self.words_path = 'words.pkl'
        self.classes_path = 'classes.pkl'
        self.specialized_topics = {
            'ingles': ['ingles', 'inglés', 'vocabulario', 'gramatica'],
            'matematicas': ['matematicas', 'matemáticas', 'mat'],
            'programacion': ['programacion', 'programación', 'codificacion', 'codigo', 'python', 'algoritmo'],
            'ciencias': ['ciencias', 'biologia', 'quimica'],
            'fisica': ['fisica', 'física', 'mecanica', 'termodinamica']
        }

    def preprocess_text(self, text: str) -> list:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()]
        return tokens

    def load_data(self, json_file: str):
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            for intent in data['intents']:
                self.classes.append(intent['tag'])
                for pattern in intent['patterns']:
                    words = self.preprocess_text(pattern)
                    self.words.extend(words)
                    self.documents.append((words, intent['tag']))
                self.intents[intent['tag']] = intent['responses']
            self.words = sorted(list(set(self.words)))
            self.classes = sorted(list(set(self.classes)))
        except Exception as e:
            st.error(f"Error al cargar los datos de entrenamiento: {e}")
            raise

    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:
            bag = [1 if word in doc[0] else 0 for word in self.words]
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = np.array([x for x, y in training])
        train_y = np.array([y for x, y in training])
        return train_x, train_y

    def build_model(self, input_shape: int, output_shape: int):
        model = Sequential([
            Dense(256, input_shape=(input_shape,), activation='relu'),
            Dropout(0.6),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(output_shape, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, training_file: str, epochs: int = 500, batch_size: int = 32):
        self.load_data(training_file)
        train_x, train_y = self.create_training_data()
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.words = pickle.load(open(self.words_path, 'rb'))
            self.classes = pickle.load(open(self.classes_path, 'rb'))
        else:
            self.model = self.build_model(len(self.words), len(self.classes))
            history = self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
            self.model.save(self.model_path)
            pickle.dump(self.words, open(self.words_path, 'wb'))
            pickle.dump(self.classes, open(self.classes_path, 'wb'))
            return history

    def is_specialized_topic(self, sentence: str) -> bool:
        words = self.preprocess_text(sentence)
        return any(word in topic_variants for topic_variants in self.specialized_topics.values() for word in words)

    def get_topic(self, sentence: str) -> str:
        words = self.preprocess_text(sentence)
        topic = next((key for key, variants in self.specialized_topics.items()
                     if any(word in variants for word in words)), None)
        return topic

    async def get_gemini_response(self, question: str):
        try:
            words = self.preprocess_text(question)
            topic = self.get_topic(question)
            if topic:
                prompt = f"Actúa como un experto en {topic} y responde: {question}"
                try:
                    response = model.generate_content(prompt)
                    if response and hasattr(response, 'text'):
                        return response.text
                    else:
                        return None
                except Exception as e:
                    st.warning(f"Error 01: {str(e)}")
                    return None
            return None
        except Exception as e:
            st.warning(f"Se ha producido un error al obtener respuesta de Gemini: {str(e)}")
            return None

    async def predict(self, sentence: str):
        if not sentence.strip():
            return "Por favor, escribe algo para que pueda ayudarte.", None, 0.0
        if self.is_specialized_topic(sentence):
            gemini_response = await self.get_gemini_response(sentence)
            if gemini_response:
                return gemini_response, self.get_topic(sentence), 0.98
        sentence_words = self.preprocess_text(sentence)
        if not sentence_words:
            return "No entiendo tu entrada. ¿Podrías reformularla?", None, 0.0
        bag = [1 if word in sentence_words else 0 for word in self.words]
        results = self.model.predict(np.array([bag]), verbose=0)[0]
        max_prob_idx = np.argmax(results)
        predicted_class = self.classes[max_prob_idx]
        probability = results[max_prob_idx]
        if probability > 0.7:
            return random.choice(self.intents[predicted_class]), self.get_topic(sentence), probability
        return "Lo siento, no entiendo tu pregunta. ¿Podrías reformularla?", None, 0.0

@st.cache_resource
def load_chatbot():
    return EbbinTutorChatbot()

def text_to_speech(text, voice_id="onwK4e9ZLuTAKqWW03F9", output_file="output.mp3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_LABS_API_KEY}
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Error al generar el audio: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")
        return False

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

def display_message(message, is_user=False, topic=None, precision=0.0):
    user_avatar_path = "User.png"
    bot_avatar_path = "Ebbin.png"
    avatar_url = get_base64_image(user_avatar_path) if is_user else get_base64_image(bot_avatar_path)
    message_class = "user" if is_user else "bot"
    timestamp = datetime.now().strftime("%H:%M:%S")
    topic_html = f'<span class="topic-chip">{topic.capitalize()}</span>' if topic else ''
    timestamp_html = f'<div class="timestamp">{timestamp}</div>' if is_user else f'<div class="timestamp">{timestamp} {topic_html}</div>' if topic_html else ''
    precision_html = f'<div class="precision">Precisión: {precision:.2%}</div>' if not is_user and precision > 0 else ''
    html = f"""
    <div class="chat-message {message_class}">
        <div class="avatar"><img src="{avatar_url}"></div>
        <div class="message">{timestamp_html}<p>{message}</p>{precision_html}</div>
    </div>
    """ if not is_user else f"""
    <div class="chat-message {message_class}">
        <div class="avatar"><img src="{avatar_url}"></div>
        <div class="message"><p>{message}</p>{timestamp_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

async def process_message(chatbot, user_input):
    display_message(user_input, True)
    with st.chat_message("assistant", avatar="Ebbin.png"):
        message_placeholder = st.empty()
        message_placeholder.markdown("_Pensando..._")
        await asyncio.sleep(1)
        response, topic, precision = await chatbot.predict(user_input)
        message_placeholder.empty()
        display_message(response, False, topic, precision)
        if st.session_state.get("play_audio", True):
            voice_id = "onwK4e9ZLuTAKqWW03F9"
            output_file = "output.wav"
            success = text_to_speech(response, voice_id, output_file)
            if success:
                audio_file = open(output_file, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"content": user_input, "is_user": True, "timestamp": datetime.now().strftime("%H:%M:%S")})
    st.session_state.messages.append({"content": response, "is_user": False, "topic": topic, "precision": precision, "timestamp": datetime.now().strftime("%H:%M:%S")})

def save_conversation():
    if "messages" in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversacion_ebbin_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for msg in st.session_state.messages:
                sender = "Tú" if msg["is_user"] else "EbbinTutor"
                f.write(f"[{msg['timestamp']}] {sender}: {msg['content']}\n")
        st.success(f"La conversación guardada como {filename}")

def display_chat_history():
    for msg in st.session_state.messages:
        precision = msg.get("precision", 0.0) if not msg["is_user"] else 0.0
        display_message(msg["content"], msg["is_user"], msg.get("topic") if not msg["is_user"] else None, precision)

def main():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "play_audio" not in st.session_state:
        st.session_state.play_audio = True
    if "audio_input" not in st.session_state:
        st.session_state.audio_input = None

    logo_base64 = get_base64_image("Logo.png")
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="{logo_base64}" style="width: 220px; height: auto;">
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("⚙️ Configuración")
    chatbot = load_chatbot()
    data_path = "training_data.json"

    if not st.session_state.initialized:
        if st.sidebar.button("Inicializar EbbinTutor"):
            with st.spinner("Inicializando EbbinTutor..."):
                chatbot.train(data_path)
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
                st.success("¡Chatbot inicializado!")
                welcome = "¡Hola! Soy EbbinTutor, tu asistente educativo. Puedo ayudarte con inglés, matemáticas, programación, ciencias y física. ¿En qué puedo ayudarte hoy?"
                st.session_state.messages.append({"content": welcome, "is_user": False, "precision": 1.0, "timestamp": datetime.now().strftime("%H:%M:%S")})

    st.sidebar.subheader("Temas especializados")
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem;">
        <span class="topic-chip">Inglés</span>
        <span class="topic-chip">Matemáticas</span>
        <span class="topic-chip">Programación</span>
        <span class="topic-chip">Ciencias</span>
        <span class="topic-chip">Física</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.subheader("Opciones de audio")
    st.session_state.play_audio = st.sidebar.checkbox("Reproducir respuestas en voz", value=st.session_state.play_audio)

    st.sidebar.subheader("Entrada de voz")
    components.html(open("recorder.html", "r", encoding="utf-8").read(), height=200)

    if st.session_state.messages and st.sidebar.button("Guardar conversación"):
        save_conversation()

    st.sidebar.markdown('<div style="height: 450px;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-footer">Created by NeuroBits</div>', unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        display_chat_history()

    if st.session_state.initialized:
        user_input = st.chat_input("Escribe tu pregunta aquí...")
        
        # Manejar entrada de voz
        if st.session_state.audio_input:
            with st.spinner("Transcribiendo audio..."):
                transcribed_text = transcribe_audio_whisper(st.session_state.audio_input)
                st.write(f"Texto transcrito: {transcribed_text}")
                if transcribed_text and transcribed_text != "No se pudo transcribir el audio.":
                    asyncio.run(process_message(st.session_state.chatbot, transcribed_text))
                else:
                    st.error("Transcripción fallida.")
                st.session_state.audio_input = None

        # Procesar entrada de texto manual
        if user_input:
            asyncio.run(process_message(st.session_state.chatbot, user_input))

        # Escuchar mensajes de JavaScript
        audio_data = st.session_state.get("audio", None)
        if audio_data:
            st.write("Audio recibido:", audio_data[:50] + "..." if len(audio_data) > 50 else audio_data)
            st.session_state.audio_input = audio_data
            del st.session_state["audio"]

    else:
        st.info("Por favor, inicializa EbbinTutor usando el botón en la barra lateral.")

    st.markdown('<div class="footer">Created by NeuroBits</div>', unsafe_allow_html=True)

def handle_audio(audio_data):
    st.session_state["audio"] = audio_data

    st.components.register("audio_recorder", handle_audio)

    if st.session_state.initialized:
        user_input = st.chat_input("Escribe tu pregunta aquí...")
        if user_input:
            asyncio.run(process_message(st.session_state.chatbot, user_input))
    else:
        st.info("Por favor, inicializa EbbinTutor usando el botón en la barra lateral.")

    st.markdown('<div class="footer">Created by NeuroBits</div>', unsafe_allow_html=True)

main()
