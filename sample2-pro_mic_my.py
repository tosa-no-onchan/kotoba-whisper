# -*- coding: utf-8 -*-
"""
kotoba-whisper/sample2-pro_mic_my.py

https://hamaruki.com/introduction-to-kotoba-whisper-a-new-option-for-japanese-speech-recognition/

copy from
  https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py
"""

import argparse
import os
import numpy as np
#import speech_recognition as sr
#import speech_recognition_my as sr
from mic_stream import MicStream
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
    parser.add_argument("--energy_threshold", default=2000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=3,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
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
    #recorder = sr.Recognizer()
    #recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    #recorder.dynamic_energy_threshold = False


    # モデルの設定
    #model_id = "./kotoba-whisper-v1.0-8bit"
    model_id = "kotoba-tech/kotoba-whisper-v1.0"

    #torch_dtype=torch.float8_e4m3fn
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    torch_dtype = torch.float32

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    generate_kwargs = {"language": "japanese", "task": "transcribe"}

    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    PIPE_USE=False
    USE_8BIT=False


    # load model
    if USE_8BIT==True:
        model_8bit = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            #quantization_config=quantization_config,
            low_cpu_mem_usage=True, 
            use_safetensors=True
            )
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True)
        model.to(device)

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
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
        )

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    #stopper=recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    mic_stream=MicStream(data_queue,
                         level_th=args.energy_threshold,
                         level_stop_th=args.energy_threshold,
                         max_sec=args.record_timeout,
                         low_sec=args.phrase_timeout
                        )
    stopper = mic_stream.start()

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
                #audio_data = b''.join(data_queue.queue)
                #data_queue.queue.clear()
                audio_data = data_queue.get()
                
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
                        if torch_dtype == torch.float16:
                            sample_half = sample.astype(np.float16)
                            input_my = torch.from_numpy(sample_half).to(device).clone()
                        else:
                            #input_my = torch.from_numpy(sample).clone()
                            input_my = torch.from_numpy(sample).to(device).clone()
                    else:
                        input_my=sample

                    # 推論の実行
                    predicted_ids = model.generate(
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

