"""
onnx2qauntize.py

https://tech.retrieva.jp/entry/20220304

https://github.com/microsoft/onnxruntime/issues/15888

quantize_dynamic(input_model, output_model, weight_type=QuantType.QInt8, nodes_to_exclude=['/conv1/Conv'])

"""

from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static

quantize_dynamic(
     model_input="kotoba-whisper-v1.0_onnx/decoder_model.onnx",
     model_output="kotoba-whisper-v1.0_onnx/decoder_model-8bit.onnx",
     weight_type=QuantType.QInt8, 
     #weight_type=QuantType.QUInt8, 
     nodes_to_exclude=['/conv1/Conv','/conv2/Conv']
)

quantize_dynamic(
     model_input="kotoba-whisper-v1.0_onnx/encoder_model.onnx",
     model_output="kotoba-whisper-v1.0_onnx/encoder_model-8bit.onnx",
     weight_type=QuantType.QInt8, 
     #weight_type=QuantType.QUInt8, 
     nodes_to_exclude=['/conv1/Conv','/conv2/Conv']
)



