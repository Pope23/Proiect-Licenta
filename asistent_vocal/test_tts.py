import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

engine.say("The result is nine")
engine.runAndWait()
