import sounddevice as sd
import numpy as np

sd.default.device = 32

fs = 48000
duration = 3

print("Înregistrare 3 secunde...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()

print("Dimensiune audio:", audio.shape)
print("Valoare maximă:", np.max(audio))
print("Valoare minimă:", np.min(audio))

