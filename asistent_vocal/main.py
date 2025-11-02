import whisper
import sounddevice as sd
import numpy as np
import sympy as sp
import pyttsx3

# ------------------------------
# 1. INIT
# ------------------------------

# Încarcă modelul Whisper (recomand: tiny sau base)
print("Încarc modelul Whisper...")
model = whisper.load_model("tiny")

# Inițializează motorul Text-to-Speech
tts = pyttsx3.init()


# ------------------------------
# 2. FUNCȚII
# ------------------------------

def record_audio(duration=4, sample_rate=16000):
    """Înregistrează vocea utilizatorului."""
    print(">> Spune ceva (înregistrare 4 secunde)...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
    sd.wait()
    return np.squeeze(recording)


def transcribe(audio):
    """Returnează textul din audio folosind Whisper."""
    print(">> Procesez vocea...")
    result = model.transcribe(audio, language="ro")
    return result["text"]


def interpret_math(text):
    """
    Traduce comanda vorbită în expresie matematică.
    Exemplu: 'doi plus trei' -> '2 + 3'
    """

    text = text.lower().strip()

    # Dicționar simplu de conversie
    replacement = {
        "plus": "+",
        "minus": "-",
        "ori": "*",
        "înmulțit cu": "*",
        "împărțit la": "/",
        "pe": "*",
        "x": "*",
        "la puterea": "**",
        "ridicat la": "**"
    }

    # Convertește cuvintele matematice
    for k, v in replacement.items():
        text = text.replace(k, v)

    # Încearcă să evalueze expresia
    try:
        expr = sp.sympify(text)
        result = expr.evalf()
        return str(result)
    except Exception:
        return None


def speak(text):
    """Vorbește răspunsul."""
    tts.say(text)
    tts.runAndWait()


# ------------------------------
# 3. LOGICA PRINCIPALĂ
# ------------------------------

def main():
    # 1. Înregistrează vocea
    audio = record_audio()

    # 2. Transcrie vocea în text
    text = transcribe(audio)
    print("Ai spus:", text)

    # 3. Încearcă să interpreteze matematic
    result = interpret_math(text)

    if result is None:
        mesaj = "Nu am putut interpreta expresia matematică."
        print(">>", mesaj)
        speak(mesaj)
    else:
        mesaj = f"Rezultatul este {result}"
        print(">>", mesaj)
        speak(mesaj)


if __name__ == "__main__":
    main()

