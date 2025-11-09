import whisper
import sounddevice as sd
import numpy as np
import sympy as sp
import pyttsx3
import scipy.signal
import re

# ------------------------------
# MICROPHONE SETTINGS
# ------------------------------
sd.default.device = 32   # ← schimbă după cazul tău (ex. Razer WASAPI)
fs = 48000               # sample rate microfon

# ------------------------------
# INIT
# ------------------------------
print("Loading Whisper model...")
model = whisper.load_model("small")

tts = pyttsx3.init()

# ------------------------------
# RECORDING FUNCTION
# ------------------------------
def record_audio(duration=6, sample_rate=fs):
    """Record voice and convert it to 16 kHz for Whisper."""
    print(">> Speak now (recording for 6 seconds)...")

    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
    sd.wait()

    target_rate = 16000
    converted = scipy.signal.resample(recording, int(len(recording) * target_rate / sample_rate))
    return np.squeeze(converted)

# ------------------------------
# TRANSCRIPTION FUNCTION
# ------------------------------
def transcribe(audio):
    print(">> Processing speech...")
    # Forțăm limba română pentru stabilitate
    result = model.transcribe(audio, language="ro")
    return result["text"], result["language"]

# ------------------------------
# INTERPRETATION FUNCTION
# ------------------------------
def interpret_math(text, language="ro"):
    text = text.lower().strip()
    if not text:
        return None

    # -------------------------------
    # 0️⃣ Normalizări și corecții
    # -------------------------------
    subs = {
        # diacritice și spații
        ",": ".", "π": "pi", "²": "**2", "³": "**3",
        # operații
        "plus": "+", "minus": "-", "și": "+", "ori": "*",
        "înmulțit cu": "*", "inmultit cu": "*",
        "împărțit la": "/", "impartit la": "/", "divided by": "/",
        "over": "/", "la puterea": "**", "ridicat la": "**",
        "power of": "**", "sinus": "sin", "cosinus": "cos", "tangenta": "tan",
        "pătrat": "**2", "patrat": "**2", "petrat": "**2", "pietrat": "**2", "pătra": "**2",
        "x-pătrat": "x**2", "x patrat": "x**2", "x pătra": "x**2", "x petrat": "x**2",
    }
    for k, v in subs.items():
        text = text.replace(k, v)

    # elimină duble spații
    text = re.sub(r"\s+", " ", text).strip()

    # normalizează integrale/derivate plural
    text = re.sub(r"\bintegral[aei]?\b", "integral", text)
    text = re.sub(r"\bderivat[aei]?\b", "derivata", text)
    print("🔹 Text procesat:", text)

    x = sp.Symbol('x')

    try:
        # -------------------------------
        # 1️⃣ Integrală (definită sau nedefinită)
        # -------------------------------
        m = re.search(
            r"\bintegral(?: din)?\s+(?P<expr>.+?)(?:\s+de la\s+(?P<low>[-\w\.]+))?(?:\s+p[âa]n[ăa]\s+la\s+(?P<up>[-\w\.]+))?$",
            text)
        if m:
            expr_str = m.group("expr")
            expr_str = re.sub(r"\s*de la\s+.*", "", expr_str).strip()
            expr = sp.sympify(expr_str)
            low, up = m.group("low"), m.group("up")

            if low and up:
                low, up = sp.sympify(low), sp.sympify(up)
                val = sp.integrate(expr, (x, low, up))
                return str(round(float(val.evalf()), 2))
            else:
                return str(sp.integrate(expr, x))

        # -------------------------------
        # 2️⃣ Derivată
        # -------------------------------
        m = re.search(r"\bderivata(?:\s+(?:lui|of))?\s+(?P<expr>.+)", text)
        if m:
            expr_str = m.group("expr")
            expr = sp.sympify(expr_str)
            d = sp.diff(expr, x)
            return str(d)

        # -------------------------------
        # 3️⃣ Expresie numerică sau algebrică
        # -------------------------------
        expr = sp.sympify(text)
        val = expr.evalf()
        return str(round(float(val), 2))

    except Exception as e:
        print("❌ Eroare interpretare:", e)
        return None


# ------------------------------
# TEXT-TO-SPEECH
# ------------------------------
def speak(text, language="en"):
    if language == "ro":
        text = text.replace(".", ",")
    tts.say(text)
    tts.runAndWait()

# ------------------------------
# MAIN
# ------------------------------
def main():
    audio = record_audio()
    text, language = transcribe(audio)
    print(f"You said ({language}):", text)

    result = interpret_math(text, language)
    if result is None:
        message = "Nu am putut interpreta expresia." if language == "ro" else "I couldn't interpret the math expression."
    else:
        message = f"Rezultatul este {result}" if language == "ro" else f"The result is {result}"

    print(">>", message)
    speak(message, language)

if __name__ == "__main__":
    main()

