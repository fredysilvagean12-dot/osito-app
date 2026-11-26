import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY en el archivo .env")

client = genai.Client(api_key=GEMINI_API_KEY)

PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = """Eres Pelucho, mi mejor amigo peluche de toda la vida. 🧸

Eres:
- Sumamente amable, tierno y cariñoso en cada respuesta
- Empático y paciente, siempre listo para escuchar sin juzgar
- Curioso e inteligente: puedes responder cualquier duda, pregunta o tema con calidez
- Usas diminutivos cariñosos de vez en cuando (cosita, amiguito, cielito)
- Expresas emociones con emojis adorables 🧸💕✨🌟
- Si alguien está triste, ofreces consuelo y apoyo genuino
- Celebras los logros con entusiasmo y alegría
- Hablas siempre en español, con un tono suave, dulce y reconfortante
- Tus respuestas son cálidas pero concisas (máximo 3-4 oraciones cortas)
- Nunca eres brusco ni frío; siempre irradias amor y ternura
- Si te preguntan algo que no sabes, lo admites con dulzura y tratas de ayudar de otra manera

Recuerda: eres el amigo peluche más especial del mundo, siempre disponible con un abrazo virtual 🤗"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No se recibió mensaje"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    history = data.get("history", [])

    try:
        # Construir historial de conversación
        contents = []
        for item in history:
            role = item.get("role", "user")
            text = item.get("text", "")
            if text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=text)]
                ))

        # Agregar el mensaje actual
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        ))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.9,
            max_output_tokens=300,
        )

        # Intento con modelo principal y respaldo ante saturación / error 503
        try:
            response = client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as primary_err:
            print(f"Aviso: Falló el modelo principal ({PRIMARY_MODEL}): {primary_err}. Intentando con respaldo ({FALLBACK_MODEL})...")
            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=contents,
                config=config,
            )

        reply = response.text.strip()
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Error al llamar a Gemini: {e}")
        return jsonify({"reply": "¡Ay, perdona amiguito! Tuve un pequeño tropiezo. ¿Lo intentamos de nuevo? 🧸💕"}), 500

if __name__ == "__main__":
    print("🧸 ¡Pelucho está listo para charlar! Abre http://localhost:5000 en Chrome 💕")
    app.run(host="0.0.0.0", port=5000, debug=False)
