"""
onnx-pred.py

https://discuss.huggingface.co/t/how-can-i-use-the-onnx-model/70923

https://huggingface.co/transformers/v4.11.3/_modules/transformers/pipelines/automatic_speech_recognition.html
"""
from transformers import AutoTokenizer, pipeline, PretrainedConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import onnxruntime
from transformers import WhisperFeatureExtractor
import time

model_id = "./kotoba-whisper-v1.0_onnx"
if False:
  encoder="encoder_model.onnx"
  decoder="decoder_model.onnx"
else:
  encoder="encoder_model-8bit.onnx"
  decoder="decoder_model-8bit.onnx"

# Load encoder model
encoder_session = onnxruntime.InferenceSession(model_id+'/'+encoder)
# Load decoder model
decoder_session = onnxruntime.InferenceSession(model_id+'/'+decoder)

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = PretrainedConfig.from_json_file(model_id+'/config.json')
#generate_kwargs = {"language": "japanese", "task": "transcribe"}

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

#model = ORTModelForSeq2SeqLM(
model = ORTModelForSpeechSeq2Seq(
    config=config,
    onnx_paths=[model_id+'/'+decoder,model_id+'/'+encoder],
    encoder_session=encoder_session, 
    decoder_session=decoder_session, 
    model_save_dir=model_id,
    use_cache=False, 
)

#onnx_translation = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer,feature_extractor=feature_extractor)
onnx_translation = pipeline(
    "automatic-speech-recognition", 
    model=model, 
    device="cpu",
    tokenizer=tokenizer)

#print('type(onnx_translation):',type(onnx_translation))
#type(onnx_translation): <class 'transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline'>
# /home/nishi/torch_env/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py

onnx_translation.feature_extractor=feature_extractor
#print('onnx_translation.feature_extractor:',pipe.feature_extractor)

sample="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

#text = 'the text to perform your translation task'
#result = onnx_translation(text, max_length = 10000)
cnt=0
start=time.time()
while True:
  result = onnx_translation(sample)
  print(result['text'])
  cnt+=1
  if cnt > 5:
    break
end=time.time()
#print("cnt:",cnt)

print("sec/f:",(end-start)/cnt)
# non quantization
# sec/f: 13.268453081448873
# quantization
# sec/f: 12.050926446914673
