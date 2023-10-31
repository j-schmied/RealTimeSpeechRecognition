"""
Real-Time Speech Recognition + Speaker Diarization

Author:     Jannik Schmied
Credits:    https://github.com/davabase/whisper_real_time
"""
import argparse
import io
import os
import speech_recognition as sr
import torch
import torchaudio
import whisper

from datetime import datetime, timedelta
from pyannote.audio import Pipeline
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


ATTENTION_STEPS = 3


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", "-e", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", "-r", default=2, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", "-t", default=2, help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)
    parser.add_argument("--print_transcript", "-v", help="Print whole transcript at the end of the live phase", action="store_true")
    parser.add_argument("--export", "-s", help="Save transcript to file", action="store_true")
    parser.add_argument("--language", "-l", help="Specify a specific language", choices=["German", "English", "French", "Spanish"], default="German")

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    print("[*]\tInitializing Audio...")

    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    sample_rate = 16000
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Prevents permanent application hang and crash by using the wrong Microphone (on Linux)
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("[i]\tAvailable microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"\t* Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=sample_rate, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=sample_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init whisper
    model = args.model
    audio_model = whisper.load_model(model, device=device)

    # Init speaker diarization
    """https://huggingface.co/pyannote/speaker-diarization"""
    sd_model = "pyannote/speaker-diarization@2.1"
    sd_pipeline = Pipeline.from_pretrained(sd_model, use_auth_token="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")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    temp_file_ext = NamedTemporaryFile().name
    
    print("[DEBUG]", temp_file, temp_file_ext)

    transcription = [""]

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print(f"[+]\tWhisper model loaded (using {args.model})")
    print(f"[+]\tSpeaker diarization model loaded (using {sd_model})")
    print(f"[+]\tUsing device: {device}")
    print("[*]\tReady, start speaking...")

    temp_file_history = list()

    while True:
        try:
            now = datetime.utcnow()

            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, "w+b") as _file:
                    data = wav_data.read()
                    
                    # Write data to temp file for transcription
                    _file.write(data)

                    # Write data to temp file list for speaker diarization
                    temp_file_history.append(data)

                if len(temp_file_history) > ATTENTION_STEPS:
                    with open(temp_file_ext, 'w+b') as _file:
                        for data in temp_file_history:
                            _file.write(data)

                    temp_file_history = temp_file_history[-ATTENTION_STEPS:]

                # Transcribe
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language=args.language)
                text = result['text'].strip()

                # Diarize the speaker
                diarization = sd_pipeline(temp_file)  # sd_pipeline(temp_file_ext)

                # If pause between recordings is detected, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Print output
                os.system('cls' if os.name=='nt' else 'clear')
                # for line in transcription:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{speaker}]", transcription[-1])
                print("", end="", flush=True)

                sleep(0.1)

        except KeyboardInterrupt:
            print("[*]\tLive mode stopped.")
            break

    if args.print_transcript:
        print("[*]\tTranscript:")
        
        for line in transcription:
            print(f"\t{line}")

    if args.export:
        transcript_path = os.path.join("..\..\output", f"transcript_session_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
        
        with open(transcript_path, "w") as f:
            for line in transcription:
                try:
                    f.write(f"{line}\n")
                except Exception as e:
                    continue

        print(f"[i]\tTranscript saved to file ({transcript_path})")


if __name__ == "__main__":
    main()
