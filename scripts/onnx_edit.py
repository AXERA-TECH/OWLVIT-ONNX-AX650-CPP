# optimum-cli export onnx --model google/owlvit-base-patch32 --task zero-shot-object-detection weights/ --opset 16

import onnx

input_path = "weights/owlvit.onnx"
output_path = "weights/model-image.onnx"
input_names = ["pixel_values"]
output_names = ["image_embeds","pred_boxes"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# output_path = "weights/model-text.onnx"
# input_names = ["input_ids","attention_mask"]
# output_names = ["/owlvit/Div_output_0"]

# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

output_path = "weights/model-post.onnx"
input_names = ["image_embeds","input_ids","/owlvit/Div_output_0"]
output_names = ["logits"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
