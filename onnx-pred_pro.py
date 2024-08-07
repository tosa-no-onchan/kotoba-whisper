"""
onnx-pred_pro.py

https://discuss.huggingface.co/t/how-can-i-use-the-onnx-model/70923

https://huggingface.co/transformers/v4.11.3/_modules/transformers/pipelines/automatic_speech_recognition.html
/home/nishi/torch_env/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py

audio_utils.ffmpeg_read
  /home/nishi/torch_env/lib/python3.10/site-packages/transformers/pipelines/audio_utils.py
https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort

"""
#from transformers import AutoTokenizer, pipeline, PretrainedConfig
from transformers import AutoTokenizer, PretrainedConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import onnxruntime
from transformers import WhisperFeatureExtractor
import time
import numpy as np
import sys,os

#from onnx import numpy_helper
import torch

# test by nishi
from transformers.pipelines.audio_utils import ffmpeg_read
#from transformers.pipelines import AutomaticSpeechRecognitionPipeline

import pyaudio


#from mltu.preprocessors import WavReader
import librosa


#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

"""
copy from /home/nishi/Documents/VisualStudio-TF/lstm_sound_to_text/inferencModel.py
"""
import matplotlib.pyplot as plt

def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
    """Plot the spectrogram of a WAV file

    Args:
        spectrogram (np.ndarray): Spectrogram of the WAV file.
        title (str, optional): Title of the plot. Defaults to None.
        transpose (bool, optional): Transpose the spectrogram. Defaults to True.
        invert (bool, optional): Invert the spectrogram. Defaults to True.
    """
    if transpose:
        spectrogram = spectrogram.T

    if invert:
        spectrogram = spectrogram[::-1]

    plt.figure(figsize=(15, 5))
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.title(f"Spectrogram: {title}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    #plt.colorbar()
    plt.tight_layout()
    plt.show()

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

if True:
  print('--------------')
  input_name0 = encoder_session.get_inputs()[0].name
  print('input_name0:',input_name0)
  # input_name: input_features
  shape0 = encoder_session.get_inputs()[0].shape
  print('shape0:',shape0)
  # shape0: ['batch_size', 128, 3000]

  outputs0 = encoder_session.get_outputs()
  print('len(outputs0):',len(outputs0))
  # len(outputs): 1
  output_name0 = encoder_session.get_outputs()[0].name
  print('output_name0:',output_name0)

  output_shape0 = encoder_session.get_outputs()[0].shape
  print('output_shape0:',output_shape0)
  output_type0 = encoder_session.get_outputs()[0].type
  print('output_type1:',output_type0)
  print('--------------')

  input_name1 = decoder_session.get_inputs()[0].name
  print('input_name1:',input_name1)
  # input_name: input_features
  shape1 = decoder_session.get_inputs()[0].shape
  print('shape1:',shape1)
  # shape: ['batch_size', 128, 3000]
  outputs1 = decoder_session.get_outputs()
  print('len(outputs1):',len(outputs1))
  # len(outputs): 1
  output_name1 = decoder_session.get_outputs()[0].name
  print('output_name1:',output_name1)
  # output_name[0]: logits
  output_shape1 = decoder_session.get_outputs()[0].shape
  print('output_shape1:',output_shape1)
  # output_shape: ['batch_size', 'decoder_sequence_length', 51866]
  output_type1 = decoder_session.get_outputs()[0].type
  print('output_type1:',output_type1)
  # output_type: tensor(float)
  print('--------------')
  
#print('type(model):',type(model))
#print('model.config:',model.config)

TEST1=False
TEST2=True
SOUND_ON=False

sample_mp3="/home/nishi/local/tmp/commonvoice/cv-corpus-11.0-2022-09-21/ja/common_voice_ja_32866812.mp3"

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

  if SOUND_ON==True:
    dx, fs = librosa.load(sample_mp3)

    #CHUNK=1024
    CHUNK=2**11
    #RATE=44100
    RATE=22050
    #RATE=16000
    p=pyaudio.PyAudio()

    stream=p.open(format = pyaudio.paInt16,
            channels = 1,
            rate = RATE,
            frames_per_buffer = CHUNK,
            input = False,
            output = True) # inputとoutputを同時にTrueにする

    #j=dx*256.0*25.0
    j=dx*2**15
    j=j.astype('int16')
    #print(j.shape)
    output = stream.write(j,j.shape[0])


if TEST2==True:
  #----------------
  # reffer from
  # https://github.com/openai/whisper
  #----------------
  import audio_my
  print('TEST2')
  audiox = audio_my.load_audio(sample_mp3,16000)
  print('audiox.shape:',audiox.shape)
  audiox = audio_my.pad_or_trim(audiox)
  print('audiox.shape:',audiox.shape)

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
  # batch axis を入れる
  sample = torch.unsqueeze(sample, 0)
  
  print('sample.shape:',sample.shape)

cnt=0
start=time.time()

while True:
  # https://medium.com/microsoftazure/build-and-deploy-fast-and-portable-speech-recognition-applications-with-onnx-runtime-and-whisper-5bf0969dd56b
  #print('model.config.forced_decoder_ids:',model.config.forced_decoder_ids)
  #print('model.config:',model.config)

  if isinstance(sample, np.ndarray):
    input_my = torch.from_numpy(sample).clone()
  else:
    input_my=sample

  #predicted_ids = model.generate(input_my, max_length=448)
  predicted_ids = model.generate(input_my)

  #print("predicted_ids:",predicted_ids)
  #x = predicted_ids.to('cpu').detach().numpy().copy()

  #print('x:shape',x.shape)
  #speech=tokenizer.decode(x[0])
  speech=tokenizer.decode(predicted_ids[0])
  print('speech:',speech)
  cnt+=1
  if cnt >= 1:
    break
end=time.time()
print("sec/f:",(end-start)/cnt)

# quantization
#sec/f: 11.72364239692688
