"""
sample2-bit8_pro_demo.py

https://hamaruki.com/introduction-to-kotoba-whisper-a-new-option-for-japanese-speech-recognition/

copy from
  https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py
"""

import argparse
import os
import numpy as np
#import speech_recognition as sr
import speech_recognition_my as sr
#import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# add by nishi
from transformers import AutoTokenizer, PretrainedConfig,AutoModelForSpeechSeq2Seq

from transformers import WhisperFeatureExtractor, AutomaticSpeechRecognitionPipeline
import time
import numpy as np
import sys,os
import librosa

from transformers.pipelines.audio_utils import ffmpeg_read

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1500,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    if False:
        model = args.model
        if args.model != "large" and not args.non_english:
            model = model + ".en"

        #audio_model = whisper.load_model(model)

    # モデルの設定
    model_id = "./kotoba-whisper-v1.0-8bit"
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch_dtype=torch.float8_e4m3fn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    generate_kwargs = {"language": "japanese", "task": "transcribe"}

    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    PIPE_USE=False

    # load model
    model_8bit = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        #quantization_config=quantization_config,
        low_cpu_mem_usage=True, 
        use_safetensors=True
        )

    #print(model_8bit)

    # tokenizer は、ローカルを使うようにすれば良いが、ここは、 huggingface から
    model_org_id="kotoba-tech/kotoba-whisper-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_org_id)
    #tokenizer = AutoTokenizer.from_pretrained(model_id)

    feature_extractor = WhisperFeatureExtractor(
        chunk_length=30,
        feature_size=128,
        hop_length=160,
        n_fft=400,
        n_samples=480000,
        nb_max_frames=3000,
        padding_side="right",
        padding_value=0.0,
        processor_class="WhisperProcessor",
        return_attention_mask=False,
        sampling_rate=16000
        )

    if PIPE_USE==True:
        # モデルのロード
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model_8bit,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
        )

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    stopper=recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    #print("Model loaded.\n")
    print("Please speak to mic.\n")


    fetch=False

    while True:
        try:
            #now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                if fetch == False:
                    print('>', end='', flush=True)

                #phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                #if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                #    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                #phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                #print('audio_np.shape:',audio_np.shape)

                # Read the transcription.
                #result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                #text = result['text'].strip()

                if PIPE_USE==True:
                    # 推論の実行
                    #result = pipe(sample_mp3, generate_kwargs=generate_kwargs)
                    result = pipe(audio_np, generate_kwargs=generate_kwargs)
                    #print(result["text"])
                    text=result["text"]

                else:
                    if True:
                        x=feature_extractor(audio_np,sampling_rate=feature_extractor.sampling_rate)
                        sample=x['input_features']
                    else:
                        import audio_my
                        audiox = audio_my.pad_or_trim(audio_np)
                        #print('audiox.shape:',audiox.shape)

                        if os.path.isfile('assets/mel_filters.npz') == False:
                            if not os.path.exists("./assets"):
                                os.makedirs("./assets")
                            np.savez_compressed(
                                "assets/mel_filters.npz",
                                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
                            )

                        #input_my = audio_my.log_mel_spectrogram(audiox,128).to("cpu")
                        sample = audio_my.log_mel_spectrogram(audiox,128)   # use mel_128
                        #print('type(sample):',type(sample))

                        # batch axis を入れる
                        sample = torch.unsqueeze(sample, 0)
                        # float32 -> float16
                        sample = sample.to(torch.float16).to(device)
                        #print('type(sample):',type(sample))


                    if isinstance(sample, np.ndarray):
                        sample_half = sample.astype(np.float16)
                        #input_my = torch.from_numpy(sample_half).clone()
                        input_my = torch.from_numpy(sample_half).to(device).clone()
                    else:
                        input_my=sample

                    # 推論の実行
                    predicted_ids = model_8bit.generate(
                        input_my,
                        None,
                        **generate_kwargs
                        )

                    #print(predicted_ids[0])
                    text=tokenizer.decode(predicted_ids[0],True)

                print(text,end='', flush=True)
                if len(text) > 0:
                    fetch=True
                    sleep(0.1)

                if False:
                    #text="hello"
                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)
            else:
                if fetch==True:
                    fetch=False
                    print('<',flush=True)
                    sleep(0.25)
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            stopper()
            break

    if False:
        print("\n\nTranscription:")
        for line in transcription:
            print(line)


if __name__ == "__main__":
    main()

