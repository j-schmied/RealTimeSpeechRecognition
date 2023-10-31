"""
Transcribe audio from initially captured audio input
"""
import datetime
import os
import pyaudio
import time
import torch
import wave
import whisper


def main():
    # Set up Whisper
    MODEL = "medium"
    model = whisper.load_model(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    # Set up audio stream
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = f"output_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    WAVE_OUTPUT_PATH = "..\..\output"

    if not os.path.exists(WAVE_OUTPUT_PATH):
        os.mkdir(WAVE_OUTPUT_PATH)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Record audio and save to file
    frames = []

    print("[*]\tStarted recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("[*]\tFinished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_file_path = os.path.join(WAVE_OUTPUT_PATH, WAVE_OUTPUT_FILENAME)

    wf = wave.open(audio_file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    start = time.time()
    result = model.transcribe(audio_file_path, fp16=torch.cuda.is_available())    
    end = time.time()
    
    transcription = result["text"].strip()
    print(f"[*]\tTranscription (took {round(end - start, 2)} s):\n\t{transcription}")


if __name__ == '__main__':
    main()
