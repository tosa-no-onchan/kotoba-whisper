# kotoba-whisper  

  Original  
  [kotoba-tech / kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)  

  kotoba-whisper-v1.0 を、 ONNX にエクスポートして試してみました。  

##### 1. How to export to ONNX    
    $ optimum-cli export onnx --model kotoba-tech/kotoba-whisper-v1.0 --task automatic-speech-recognition kotoba-whisper-v1.0_onnx/   

##### 2. Run ONNX    
    $ python onnx-pred.py  

##### 3. ONNX quantization  
    $ python onnx2qauntize.py  

##### 4. Run ONNX quantize  
    $ python onnx-pred_pro.py  
