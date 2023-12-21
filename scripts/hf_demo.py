import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "./test.jpg"
image = Image.open(url)
texts = [["a photo of people"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

torch.onnx.export(model, (inputs["input_ids"],inputs["pixel_values"],inputs["attention_mask"]), "weights/owlvit.onnx", opset_version=14,
                  input_names=["input_ids","pixel_values","attention_mask"],
                  output_names=["logits","pred_boxes","text_embeds","image_embeds"])

print(inputs)
# print(processor)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")