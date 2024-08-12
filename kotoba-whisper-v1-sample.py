""""
  kotoba-whisper-v1-sample.py
https://dev.classmethod.jp/articles/kotoba-whisper-v1-0/

"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio

import time

# config
model_id = "kotoba-tech/kotoba-whisper-v1.0"
#torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch_dtype = torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
#model_kwargs = {"attn_implementation": "flash_attention_2"} if torch.cuda.is_available() else {}

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
    model_kwargs=model_kwargs
)

if False:
    # load sample audio & downsample to 16kHz
    dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0]["audio"]

else:
    sample="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

cnt=0
start=time.time()
while True:

    # run inference
    result = pipe(sample)
    print(result["text"])
    cnt+=1
    if cnt >= 5:
        break

end=time.time()
#print("cnt:",cnt)

print("sec/f:",(end-start)/cnt)

# sec/f: 7.274289846420288
