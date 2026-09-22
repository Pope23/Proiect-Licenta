import whisper
import sounddevice as sd
import numpy as np
import sympy as sp
import subprocess
import scipy.signal
import re

fs = 48000

def setup_audio_devices():
    devices = sd.query_devices()

    input_device = None
    output_device = None

    for i, device in enumerate(devices):
        name = device["name"]

        if "C170" in name and device["max_input_channels"] > 0:
            input_device = i

        if device["max_output_channels"] > 0 and output_device is None:
            output_device = i

    if input_device is None:
        raise RuntimeError("Microphone C170 was not found.")

    sd.default.device = input_device

    print("Input device:", input_device, devices[input_device]["name"])

    if output_device is not None:
        print("Output device:", output_device, devices[output_device]["name"])
    else:
        print("Output device: none detected by sounddevice")

setup_audio_devices()

print("Loading Whisper model...")
model = whisper.load_model("base")


x = sp.Symbol("x")

locals_dict = {
    "x": x,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "e": sp.E
}

VOCAB = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied": "*",
    "multiply": "*",
    "over": "/",
    "divided": "/",
    "divide": "/",

    "squared": "**2",
    "square": "**2",
    "cubed": "**3",
    "cube": "**3",

    "sin": "sin",
    "sine": "sin",
    "cos": "cos",
    "cosine": "cos",
    "tan": "tan",
    "tangent": "tan",
    "log": "log",
    "sqrt": "sqrt",
    "root": "sqrt",

    "open": "(",
    "opened": "(",
    "close": ")",
    "closed": ")",
    "parenthesis": "",
    "parentheses": "",
    "bracket": "",
    "brackets": "",
    "per": "/",

    "x": "x",
    "pi": "pi",
    "e": "e"
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "millions": "1000000"    
}

IGNORE_WORDS = {
    "the",
    "a",
    "an",
    "by",
    "with",
    "please",
    "will",
    "would",
    "is",
    "calculate",
    "compute",
    "find"
}

def record_audio(sample_rate=fs, max_duration=10, silence_duration=1.4, threshold=0.012):
    print(">> Speak now...")

    audio_chunks = []
    silent_chunks = 0

    chunk_duration = 0.2
    chunk_samples = int(chunk_duration * sample_rate)

    max_chunks = int(max_duration / chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)

    started_speaking = False

    for _ in range(max_chunks):

        chunk = sd.rec(
            chunk_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        volume = np.sqrt(np.mean(chunk ** 2))

        if volume > threshold:
            started_speaking = True
            silent_chunks = 0
            audio_chunks.append(chunk)

        else:
            if started_speaking:
                silent_chunks += 1
                audio_chunks.append(chunk)

                if silent_chunks >= silence_chunks_needed:
                    break

    if not audio_chunks:
        return np.array([], dtype="float32")

    recording = np.concatenate(audio_chunks, axis=0)

    target_rate = 16000

    converted = scipy.signal.resample(
        recording,
        int(len(recording) * target_rate / sample_rate)
    )

    return np.squeeze(converted)

def transcribe(audio):
    print(">> Processing speech...")

    result = model.transcribe(
        audio,
        language="en",
        fp16=False
    )

    return result["text"], result["language"]

def autocorrect(text):
    text = text.lower()

    fixes = {
        "integral of": "integral",
        "integrals": "integral",
        "integrate": "integral",

        "derivative of": "derivative",
        "differentiate": "derivative",
        
        "x squared": "x power 2",
        "x cubed": "x power 3",
        "x square": "x squared",
        "x to the power of two": "x squared",
        "x to the power of 2": "x squared",
        "x power two": "x squared",
        "x power 2": "x squared",

        "x cube": "x cubed",
        "x to the power of three": "x cubed",
        "x to the power of 3": "x cubed",
        "x power three": "x cubed",
        "x power 3": "x cubed",

        "opened parenthesis": "open parenthesis",
        "closed parenthesis": "close parenthesis",
        "open parentheses": "open parenthesis",
        "close parentheses": "close parenthesis",

        "divided by": "over",
        "multiplied by": "times",

        "square root": "sqrt",

        "sine": "sin",
        "cosine": "cos",
        "tangent": "tan",

        "comma": "",
        "period": "",
        ".": "",
        ",": ""
    }
    
    text = re.sub(r"x to the power of (\d+)", r"x power \1", text)
    text = re.sub(r"x to the power of one", "x power 1", text)
    text = re.sub(r"x to the power of two", "x power 2", text)
    text = re.sub(r"x to the power of three", "x power 3", text)
    text = re.sub(r"x to the power of four", "x power 4", text)
    text = re.sub(r"x to the power of five", "x power 5", text)
    text = re.sub(r"(\d+)\s+million", r"\1 * 1000000", text)
    text = re.sub(r"(\d+)\s+millions", r"\1 * 1000000", text)
    text = re.sub(r"(\d+)\s+thousand", r"\1 * 1000", text)

    for k, v in fixes.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_text(text):
    text = autocorrect(text)

    text = text.replace("^", " power ")
    text = text.replace("(", " open ")
    text = text.replace(")", " close ")
    text = text.replace("+", " plus ")
    text = text.replace("-", " minus ")
    text = text.replace("*", " times ")
    text = text.replace("/", " over ")

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text):
    text = normalize_text(text)

    words = text.split()

    tokens = []
    i = 0

    while i < len(words):
        w = words[i]

        if w in IGNORE_WORDS:
            i += 1
            continue

        if w in ["integral", "integrals", "integrate"]:
            tokens.append("INTEGRAL")
            i += 1
            continue

        if w in ["derivative", "differentiate"]:
            tokens.append("DERIVATIVE")
            i += 1
            continue

        if w == "from":
            tokens.append("FROM")
            i += 1
            continue

        if w == "to":
            tokens.append("TO")
            i += 1
            continue

        if w == "of":
            tokens.append("OF")
            i += 1
            continue

        if w == "power":
            if i + 1 < len(words):

                nxt = words[i + 1]

                if nxt in NUMBER_WORDS:
                    tokens.append("**")
                    tokens.append(NUMBER_WORDS[nxt])
                    i += 2
                    continue

                if re.fullmatch(r"\d+", nxt):
                    tokens.append("**")
                    tokens.append(nxt)
                    i += 2
                    continue

        if w in NUMBER_WORDS:
            tokens.append(NUMBER_WORDS[w])
            i += 1
            continue

        if re.fullmatch(r"\d+", w):
            tokens.append(w)
            i += 1
            continue

        if re.fullmatch(r"\d+x", w):
            tokens.append(w[:-1])
            tokens.append("*")
            tokens.append("x")
            i += 1
            continue

        if w in VOCAB:
            token = VOCAB[w]

            if token:
                tokens.append(token)

            i += 1
            continue

        i += 1

    return tokens

def build_expression(tokens):

    allowed = {
        "x", "+", "-", "*", "/", "(", ")", "**", "**2", "**3",
        "sin", "cos", "tan", "log", "sqrt", "pi", "e"
    }

    expr = []

    for token in tokens:

        if token in allowed:
            expr.append(token)

        elif re.fullmatch(r"\d+", token):
            expr.append(token)

    expression = " ".join(expr)

    expression = expression.replace("x **2", "x**2")
    expression = expression.replace("x **3", "x**3")

    expression = expression.replace("** 2", "**2")
    expression = expression.replace("** 3", "**3")

    expression = re.sub(r"(\d)\s+x", r"\1*x", expression)
    expression = re.sub(r"x\s+(\d)", r"x*\1", expression)

    expression = re.sub(r"(\d)\s+\(", r"\1*(", expression)
    expression = re.sub(r"\)\s+(\d)", r")*\1", expression)

    expression = re.sub(r"x\s+\(", r"x*(", expression)
    expression = re.sub(r"\)\s+x", r")*x", expression)

    expression = expression.replace("sin (", "sin(")
    expression = expression.replace("cos (", "cos(")
    expression = expression.replace("tan (", "tan(")
    expression = expression.replace("log (", "log(")
    expression = expression.replace("sqrt (", "sqrt(")

    expression = re.sub(r"\s+", " ", expression).strip()

    if not expression:
        raise ValueError("Empty expression")

    return expression

def parse_integral(tokens):

    if "FROM" in tokens and "TO" in tokens and "OF" in tokens:

        idx_from = tokens.index("FROM")
        idx_to = tokens.index("TO")
        idx_of = tokens.index("OF")

        low = tokens[idx_from + 1]
        up = tokens[idx_to + 1]

        expr_tokens = tokens[idx_of + 1:]

        expr_string = build_expression(expr_tokens)

        print("Expr:", expr_string)

        expr = sp.sympify(expr_string, locals=locals_dict)

        return sp.integrate(
            expr,
            (x, float(low), float(up))
        )

    if "OF" in tokens:
        idx_of = tokens.index("OF")
        expr_tokens = tokens[idx_of + 1:]

    else:
        expr_tokens = tokens[1:]

    expr_string = build_expression(expr_tokens)

    print("Expr:", expr_string)

    expr = sp.sympify(expr_string, locals=locals_dict)

    return sp.integrate(expr, x)

def parse_derivative(tokens):

    if "OF" in tokens:
        idx_of = tokens.index("OF")
        expr_tokens = tokens[idx_of + 1:]

    else:
        expr_tokens = tokens[1:]

    expr_string = build_expression(expr_tokens)

    print("Expr:", expr_string)

    expr = sp.sympify(expr_string, locals=locals_dict)

    return sp.diff(expr, x)

def parse_expression(tokens):

    expr_string = build_expression(tokens)

    print("Expr:", expr_string)

    expr = sp.sympify(expr_string, locals=locals_dict)

    return expr.evalf()

def format_result(result):
    result = sp.N(result)

    try:
        value = float(result)

        if value.is_integer():
            return str(int(value))

        return f"{value:.2f}".rstrip("0").rstrip(".")

    except Exception:
        return str(result)

def interpret_math(text):

    tokens = tokenize(text)

    print("Tokens:", tokens)

    try:

        if not tokens:
            return None

        if "INTEGRAL" in tokens or (
            "FROM" in tokens and
            "TO" in tokens and
            "OF" in tokens
        ):
            result = parse_integral(tokens)
            if result.free_symbols:
                return str(result)
                
            return format_result(result)

        if "DERIVATIVE" in tokens:
            result = parse_derivative(tokens)
            return str(result)

        result = parse_expression(tokens)

        return format_result(result)

    except Exception as e:
        print("Eroare interpretare:", e)
        return None

def format_spoken_result(result):
    text = str(result)

    text = text.replace("**", " to the power of ")
    text = text.replace("*", " times ")
    text = text.replace("/", " divided by ")
    text = text.replace("+", " plus ")
    text = text.replace("-", " minus ")

    text = re.sub(r"\s+", " ", text).strip()

    return text

def speak(text):

    subprocess.run(["espeak", str(text)], check=False)

def main():
    speak("Math assistant started.")

    while True:
        audio = record_audio()

        if audio.size == 0:
            continue

        text, language = transcribe(audio)

        print(f"You said ({language}): {text}")

        text_lower = text.lower()

        wake_words = [
            "math assistant",
            "assistant",
            "calculate",
            "compute",
            "integral",
            "derivative",
            "solve"
        ]

        if not any(word in text_lower for word in wake_words):
            print(">> Ignored. No math request detected.")
            continue

        result = interpret_math(text)

        if result is None:
            message = "I could not interpret the expression."
        else:
            spoken_result = format_spoken_result(result)
            message = f"The result is {spoken_result}"

        print(">>", message)
        speak(message)
if __name__ == "__main__":
    main()
