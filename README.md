# kotoba-whisper  

  Original  
  [kotoba-tech / kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0)  

  kotoba-whisper-v1.0 を、 ONNX にエクスポートして試してみました。  

  おんちゃんのブログに詳しく書いています。  
  [transformer asr japanese サンプルがある。kotoba-whisper-v1.0 を ONNX に変換](http://www.netosa.com/blog/2024/07/transformer-asr-japanese.html)  

##### 1. How to export to ONNX    
    $ optimum-cli export onnx --model kotoba-tech/kotoba-whisper-v1.0 --task automatic-speech-recognition kotoba-whisper-v1.0_onnx/   

##### 2. Run ONNX    
    $ python onnx-pred.py  

##### 3. ONNX quantization  
    $ python onnx2qauntize.py  

##### 4. Run ONNX predict without pipelines  
    $ python onnx-pred_pro.py  

##### 5. Run Original GPU model  
    $ python kotoba-whisper-v1-sample.py  
    $ python sample2-pro.py  

##### 6. Run Real Time MIC Input with GPU model  
    $ python sample2-pro_mic_my.py  

##### 7. Run Real Time MIC Input ONNX  
    $ python onnx_pred_pro_mic_my.py  

##### 8. MIC Input Stream Class.  
mic_stream.py  
``````
   # using for class  
    from mic_stream import MicStream  
    from queue import Queue  
    data_queue = Queue()

    mic_stream=MicStream(data_queue,level_th=3500.0,level_stop_th=3500.0,max_sec=3,low_sec=1.5)
    stopper = mic_stream.start()

    while True:
        try:
            if not data_queue.empty():
                #print('get queue')
                audio_data = data_queue.get()

        except KeyboardInterrupt:
            stopper()
            break

``````    
    
