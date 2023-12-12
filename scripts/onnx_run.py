import onnxruntime

class onnx_inferencer:

    def __init__(self, model_path) -> None:
        self.onnx_model_sess = onnxruntime.InferenceSession(model_path)
        self.output_names = []
        self.input_names = []
        print(model_path)
        for i in range(len(self.onnx_model_sess.get_inputs())):
            self.input_names.append(self.onnx_model_sess.get_inputs()[i].name)
            print("    input:", i,
                  self.onnx_model_sess.get_inputs()[i].name,self.onnx_model_sess.get_inputs()[i].type,
                  self.onnx_model_sess.get_inputs()[i].shape)

        for i in range(len(self.onnx_model_sess.get_outputs())):
            self.output_names.append(
                self.onnx_model_sess.get_outputs()[i].name)
            print("    output:", i,
                  self.onnx_model_sess.get_outputs()[i].name,self.onnx_model_sess.get_outputs()[i].type,
                  self.onnx_model_sess.get_outputs()[i].shape)
        print("")

    def get_input_count(self):
        return len(self.input_names)

    def get_input_shape(self, idx: int):
        return self.onnx_model_sess.get_inputs()[idx].shape

    def get_input_names(self):
        return self.input_names

    def get_output_count(self):
        return len(self.output_names)

    def get_output_shape(self, idx: int):
        return self.onnx_model_sess.get_outputs()[idx].shape

    def get_output_names(self):
        return self.output_names

    def inference(self, tensor):
        return self.onnx_model_sess.run(
            self.output_names, input_feed={self.input_names[0]: tensor})

    def inference_multi_input(self, tensors: list):
        inputs = dict()
        for idx, tensor in enumerate(tensors):
            inputs[self.input_names[idx]] = tensor
        return self.onnx_model_sess.run(self.output_names, input_feed=inputs)

backbone = onnx_inferencer("weights/owlvit-image.onnx")
bert = onnx_inferencer("weights/owlvit-text.onnx")
transformer = onnx_inferencer("weights/owlvit-post.onnx")

import torchvision.transforms as T
from tokenizer import build_tokenizer
import torch
import cv2
import numpy as np
from PIL import Image

def load_image(image_path: str):
    transform = T.Compose(
        [
            T.Resize([768,768]),
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954,0.26130258,0.27577711]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


tokenizer = build_tokenizer()

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

IMAGE_PATH = "./test.jpg"
TEXT_PROMPT = ["football"]
image_source, image = load_image(IMAGE_PATH)
print(image.shape)
image_embeds, pred_boxes = backbone.inference(image.unsqueeze(0).numpy())

print(image_embeds[0].shape)

input_ids = np.array([tokenizer.encode(t) for t in TEXT_PROMPT]).reshape(-1)
print(input_ids)


input_ids = np.pad([49406,*input_ids,49407],(0,16-len(input_ids)-2))
print(input_ids)
mask = (input_ids > 0).astype(np.int64)

print(mask)

text_embeds = bert.inference_multi_input([input_ids.reshape(1,16), mask.reshape(1,16)])[0].reshape(1,-1)
print(text_embeds)
logits = transformer.inference_multi_input([image_embeds[0].reshape(1,24,24,768),text_embeds,input_ids.reshape(1,16)])[0]

logits = torch.Tensor(logits).sigmoid().numpy().reshape(-1)
pred_boxes = pred_boxes.reshape(-1,4)

print(logits.shape)
print(pred_boxes.shape)

# get idx of boxes with confidence > BOX_TRESHOLD
idxs = np.where(logits > BOX_TRESHOLD)[0]
if(len(idxs) == 0):
    print("no boxes found")
    exit()
print(idxs)
# print(logits[idx])
# print(pred_boxes[idx][0])
_h,_w,_ = image_source.shape
image_source = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
for idx in idxs:
    print(idx,pred_boxes[idx])
    xc,yc,w,h = pred_boxes[idx]
    xc*=_w
    yc*=_h
    w*=_w
    h*=_h

    
    cv2.rectangle(image_source,(int(xc-w/2),int(yc-h/2)),(int(xc+w/2),int(yc+h/2)),(0,0,255),2)
cv2.imwrite("out.jpg", image_source)

