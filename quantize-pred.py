"""

kotoba-whisper/quantize-pred.py

https://huggingface.co/docs/transformers/quantization/bitsandbytes
https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model

"""
import sys
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, BitsAndBytesConfig,AutoModelForSpeechSeq2Seq
from transformers import WhisperFeatureExtractor

import time

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "kotoba-tech/kotoba-whisper-v1.0"
generate_kwargs = {"language": "japanese", "task": "transcribe"}

#model_8bit = AutoModelForCausalLM.from_pretrained(
model_8bit = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
	low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

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

pipe = pipeline(
    "automatic-speech-recognition", 
    model=model_8bit, 
    torch_dtype=torch.float8_e4m3fn,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor
)
sample="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

#print('type(model_8bit):',type(model_8bit))
"""
type(model_8bit): <class 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'>
/home/nishi/torch_env/lib/python3.10/site-packages/transformers/models/whisper/generation_whisper.py:480: FutureWarning: 
The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
"""
#print('model_8bit.input_features:',model_8bit.input_features)

if False:
    attributes = dir(model_8bit)
    for attribute in attributes:
        print(attribute)

model_8bit.generation_config.forced_decoder_ids=None

cnt=0
start=time.time()
while True:
    # 推論の実行
    result = pipe(sample, generate_kwargs=generate_kwargs)
    #result = pipe(sample)
    print(result["text"])
    cnt+=1
    if cnt > 5:
        break

end=time.time()
#print("cnt:",cnt)

print("sec/f:",(end-start)/cnt)
# sec/f: 3.9915151596069336

#print('model_8bit.generation_config:',model_8bit.generation_config)

#model.push_to_hub("bloom-560m-8bit")
#model.save_pretrained("path/to/model")
if False:
    model_8bit.save_pretrained("./kotoba-whisper-v1.0-8bit")
#print(model)



