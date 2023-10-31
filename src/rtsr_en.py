"""
Working PoC, only for English language (model restriction)
Using AssemblyAI API + Streamlit
"""
import asyncio
import base64
import datetime
import dotenv
import json
import pyaudio
import streamlit as st
import websockets


env = dotenv.dotenv_values()

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
API_KEY = env["ASSEMBLYAI_API_KEY"]


async def send_receive(stream):
    print(f"[*] Connecting to {URL}")

    async with websockets.connect(
            URL,
            extra_headers=(("Authorization", API_KEY),),
            ping_interval=5,
            ping_timeout=20) as _ws:

        r = await asyncio.sleep(0.1)

        print("[*] Receiving SessionBegins")

        session_begins = await _ws.recv()
        print("[*]", session_begins)
        print("[*] Sending messages")

        async def send():
            while st.session_state["run"]:
                try:
                    data = stream.read(FRAMES_PER_BUFFER)
                    data = base64.b64encode(data).decode("utf-8")
                    json_data = json.dumps({"audio_data": str(data)})
                    r = await _ws.send(json_data)

                except websockets.exceptions.ConnectionClosed as e:
                    print("[*] Connection closed:", e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    assert False, "Not a websocket 4008 error"

                r = await asyncio.sleep(0.01)

            return True

        async def receive():
            while st.session_state["run"]:
                try:
                    result_str = await _ws.recv()

                    if json.loads(result_str)["message_type"] == "FinalTranscript":
                        transcript = json.loads(result_str)["text"]
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {transcript}")
                        st.markdown(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {transcript}")

                except websockets.exceptions.ConnectionClosed as e:
                    print("[*] Connection closed:", e)
                    assert e.code == 4008
                    break

                except Exception as e:
                    assert False, "Not a websocket 4008 error"

        send_result, receive_result = await asyncio.gather(send(), receive())


def main():
    # Initialize audio stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    # Initialize streamlit
    if "run" not in st.session_state:
        st.session_state["run"] = False

    st.title("Real-Time Speech-Recognition")
    
    def st_start_listening():
        st.session_state["run"] = True
    
    def st_stop_listening():
        st.session_state["run"] = False
    
    start, stop = st.columns(2)
    start.button("Start listening", on_click=st_start_listening)
    stop.button("Stop listening", on_click=st_stop_listening)

    # Start transcribing
    asyncio.run(send_receive(stream))


if __name__ == "__main__":
    main()

