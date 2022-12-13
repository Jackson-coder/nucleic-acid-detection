import torch
import torch.nn as nn
import onnx
import numpy as np
from src.body import Body
from src.hand import Hand
import onnxruntime as ort
from onnxsim import simplify
import time
from tqdm import tqdm
from torch_utils import time_synchronized 

from onnxmltools.utils import float16_converter

#184 1 2 3 4
body_estimation = Hand('model/hand_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')
# 若存在batchnorm、dropout层则一定要eval()!!!!再export
body_estimation = body_estimation.model
# hand_estimation = hand_estimation.model

input_name = ["input"]
output_name = ["output"]

# x=torch.randn((1,3,184,328)).float().cuda()
x=torch.randn((1,3,736,736)).float().cuda()

print("building onnx file")
torch.onnx.export(body_estimation,x,'body.onnx',input_names=input_name,output_names=output_name)
# torch.onnx.export(hand_estimation,x,'hand.onnx',input_names=input_name,output_names=output_name)
print("building onnx file finished")
#简化onnx
body_model = onnx.load('body.onnx')
# hand_model = onnx.load('hand.onnx')
body_model_simp, check1 = simplify(body_model)
# hand_model_simp, check2 = simplify(hand_model)
assert check1, "Simplified ONNX model could not be validated"
# assert check2, "Simplified ONNX model could not be validated"
onnx.save(body_model_simp, 'body.onnx')
# onnx.save(hand_model_simp, 'hand.onnx')
# onnx.save(body_model, 'body.onnx')
print("the onnx file has been simplified")

# ort_session = ort.InferenceSession(onnx_filepath, providers=['CUDAExecutionProvider','TensorrtExecutionProvider'])#'TensorrtExecutionProvider', 
ort_session = ort.InferenceSession('body.onnx', providers=['CUDAExecutionProvider'])
print(ort.get_device())

# 测试输出

loop = 200
total_time = 0
for i in tqdm(range(loop)):
    time_start = time_synchronized()
    with torch.no_grad():
        output = body_estimation(x)
    time_end = time_synchronized()
    # print("time : {}".format(time_end - time_start))
    if i>0:
        total_time += (time_end - time_start)
pytorch_time = total_time / (loop-1)
print("\n body torch avg time : {}".format(total_time / (loop-1)))


loop = 200
total_time = 0
for i in tqdm(range(loop)):
    time_start = time_synchronized()
    features = ort_session.run(None, {'input': x.cpu().numpy()})[0]
    time_end = time_synchronized()
    # print("time : {}".format(time_end - time_start))
    if i>0:
        total_time += (time_end - time_start)
pytorch_time = total_time / (loop-1)
print("\n body onnx avg time : {}".format(total_time / (loop-1)))

