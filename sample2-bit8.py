"""
 sample2-bit8.py

https://hamaruki.com/introduction-to-kotoba-whisper-a-new-option-for-japanese-speech-recognition/
"""
import torch
from transformers import pipeline,AutoModelForSpeechSeq2Seq,BitsAndBytesConfig,AutoTokenizer
from transformers import AutomaticSpeechRecognitionPipeline
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor


import sys
import time

from transformers.pipelines.audio_utils import ffmpeg_read


# モデルの設定
model_id = "./kotoba-whisper-v1.0-8bit"
#torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch_dtype=torch.float8_e4m3fn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "japanese", "task": "transcribe"}


#quantization_config = BitsAndBytesConfig(load_in_8bit=True)

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


# モデルのロード
if False:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_8bit,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )
else:
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model_8bit,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
    )

"""
model.model.generate() 使えるみたい。

/home/nishi/torch_env/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py
            tokens = self.model.generate(
                inputs=inputs,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

"""

if False:
    # サンプル音声を読み込み(16kHzにダウンサンプリング)
    dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
    sample = dataset[0]["audio"]

else:
    #result = pipe("audio.mp3", generate_kwargs=generate_kwargs)
    sample_mp3="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

#print('pipe.feature_extractor:',pipe.feature_extractor)


TEST1=True

if TEST1==True:
  print('TEST1')
  with open(sample_mp3, "rb") as f:
    inputs = f.read()
  inputs = ffmpeg_read(inputs, sampling_rate=feature_extractor.sampling_rate)
  print('inputs.shape:',inputs.shape)
  
  x=feature_extractor(inputs,sampling_rate=feature_extractor.sampling_rate)
  #print('type(x):',type(x))
  #x_dict=dict(x)
  #print(x_dict)
  dt=x['input_features']
  #print('dt.shape',dt.shape)
  sample=dt
  if False:
    dtx=np.transpose(dt[0])
    plot_spectrogram(dtx)
  print('sample.shape',sample.shape)

cnt=0
start=time.time()
while True:
    # 推論の実行
    result = pipe(inputs, generate_kwargs=generate_kwargs)
    print(result["text"])
    cnt+=1
    if cnt >= 3:
        break

end=time.time()
#print("cnt:",cnt)

print("sec/f:",(end-start)/cnt)
# sec/f: 4.035113255182902
