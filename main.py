import asyncio
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
from pydub import AudioSegment
from pydub.playback import play
from faster_whisper import WhisperModel
from gemini_chat import ask_gemini
import edge_tts

# Load Whisper model once
model = WhisperModel("base", compute_type="int8")

def recognize_speech() -> str:
    print("🎤 Recording... Speak now.")
    duration = 5  # seconds
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, fs, recording)
        print("🧠 Transcribing with faster-whisper...")
        segments, _ = model.transcribe(f.name)
        return " ".join([segment.text for segment in segments])

async def speak_text(text: str):
    import edge_tts
    import os
    from pydub import AudioSegment
    from pydub.playback import play

    output_file = "response.mp3"

    print(f"🤖 Gemini: {text}")

    # Generate speech
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)

    # Play it
    sound = AudioSegment.from_file(output_file, format="mp3")
    play(sound)

    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)


if __name__ == "__main__":
    while True:
        user_text = recognize_speech().strip()
        if not user_text:
            print("❌ Could not understand. Try again.")
            continue
        if user_text.lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting assistant. Goodbye!")
            break

        reply = ask_gemini(user_text)
        asyncio.run(speak_text(reply))
