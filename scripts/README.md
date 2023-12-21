# ONNX Export

## Run Huggingface Demo
you will get the `owlvit.onnx` model file in ./weights
``` bash
mkdir weights
python hf_demo.py
```

## ONNX Simpifier
``` bash
pip install onnxsim
onnxsim weights/owlvit.onnx weights/owlvit.onnx
```

## Get Sub Graph ONNX Model
maybe you need to modify the input output node name in the python script,\
and you will get 3 sub models in weights
``` bash
python onnx_edit.py
tree weights
```
get log like this
```bash
weights/
├── owlvit-image.onnx  // for image encode
├── owlvit-post.onnx   // for text encode
├── owlvit-text.onnx   // post process to get logits result
└── owlvit.onnx        // original model export from huggingface demo
```


## Modify ONNX IO Name in Code
[Here](../src/Runner/OWLVIT.hpp#L25)
```Cpp
    const char
        *TextEncInputNames[2]{"input_ids", "attention_mask"},
        *TextEncOutputNames[1]{"text_embeds"},
        *DecoderInputNames[3]{"image_embeds", "/owlvit/Div_output_0", "input_ids"},
        *DecoderOutputNames[1]{"logits"};
```