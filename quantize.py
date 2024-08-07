"""

kotoba-whisper/quantize.py

https://huggingface.co/docs/transformers/quantization/bitsandbytes
https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model

"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoModelForSpeechSeq2Seq, GenerationConfig, TextStreamer

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

#model = AutoModelForCausalLM.from_pretrained(
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "kotoba-tech/kotoba-whisper-v1.0", 
    quantization_config=quantization_config,
	low_cpu_mem_usage=True
)


#generation_config = GenerationConfig(
#    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
#)


tokenizer = AutoTokenizer.from_pretrained("kotoba-tech/kotoba-whisper-v1.0")

# inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
#inputs = tokenizer(["'max_length': 448, 'begin_suppress_tokens': [220, 50257]"], return_tensors="pt")

#streamer = TextStreamer(tokenizer)
#_ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)

#model.push_to_hub("bloom-560m-8bit")

#model.save_pretrained("path/to/model")
model.save_pretrained("./kotoba-whisper-v1.0-8bit")
#print(model)
