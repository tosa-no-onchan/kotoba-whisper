"""
https://hamaruki.com/introduction-to-kotoba-whisper-a-new-option-for-japanese-speech-recognition/
"""
import torch
from transformers import pipeline,AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset, Audio

import time

# モデルの設定
model_id = "kotoba-tech/kotoba-whisper-v1.0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "japanese", "task": "transcribe"}

# load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
    #model_kwargs=model_kwargs
)
if False:
    # サンプル音声を読み込み(16kHzにダウンサンプリング)
    dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
    sample = dataset[0]["audio"]

else:
    #result = pipe("audio.mp3", generate_kwargs=generate_kwargs)
    sample="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

#print('pipe.feature_extractor:',pipe.feature_extractor)
"""
pipe.feature_extractor: WhisperFeatureExtractor {
  "chunk_length": 30,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 128,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "nb_max_frames": 3000,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "WhisperProcessor",
  "return_attention_mask": false,
  "sampling_rate": 16000
}
"""

cnt=0
start=time.time()
while True:
    # 推論の実行
    result = pipe(sample, generate_kwargs=generate_kwargs)
    print(result["text"])

    cnt+=1
    if cnt > 5:
        break

end=time.time()
#print("cnt:",cnt)

print("sec/f:",(end-start)/cnt)
# sec/f: 3.8781877358754477
