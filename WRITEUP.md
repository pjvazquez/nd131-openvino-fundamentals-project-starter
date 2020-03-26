# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## link to the model and how to convert it to an IR

The model I used is a darknet YOLO model trained with the COCO dataset I dowloaded from Intel
https://github.com/mystic123/tensorflow-yolo-v3  GitHub repository (commit ed60b90)
I obtained yolov3.weights file, that using the weights_pb.py script I transformed in a frozen_darknet_yolov3_model.pb
Then using the script: 

python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --batch 1

I transformed the YOLO weights to its IR

All this code is in the project repository under the tensorflow-yolo-v3 folder.


## Explaining Custom Layers

One of the reasons of the success of deep learning is the wide range of layers that can be used an combined to create a model. But this is because developers are not constrained to the existent layers, but can create their own ones so they can solve an specific problem or implement an specific operation you can need in your model, or creating a customized version of an existing layer.

To create a custom layer for OpenVino, you must add extensions to the Model Optimizer and the Inference Engine.
For this, the first step is to use the Model Extension Generator tool
The MEG is going to create templates for Model Optimizer extractor extension, Model Optimizer operations extension, Inference Engine CPU extension and Inference Engine GPU extension.
Once customized the templates, next step is to generate the IR files with the Model Optimizer.
Finally, before using the custom layer in your model with the Inference Engine, you must: first, edit the CPU extension template files, second, compile the CPU extension and finally, execute the model with the custom layer.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were:
- Processing speed: comparing processing speed in the same environment is usefull to see if model performs faster after transformation in IR. In my case, the YOLOv3 model 
- Model accuracy
- Hardware needs

The difference between model accuracy pre- and post-conversion was relatively low

The size of the model pre- and post-conversion was similar

The inference time of the model pre- and post-conversion was similar.

The model needs less hardware than the original one for a similar performance

## Assess Model Use Cases

Some of the potential use cases of the people counter app are of interest for retail commerce to know how and when customers get into the store or are in some point of interest. It’s also usefull for security control, measuring how people performs in restricted or controlled spaces.

Each of these use cases would be useful because allows to improve marketing and control strategies both in the retail store or in the security control.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. 
Lighting, focal length a,d image size are relevant to system behavior, a bad lighting can decrease model performance by diffusing image info, image size is relevant although YOLO models are quite size independent and the same happens with focal length.
Camera vision angle is also relevant for this kind of tasks and for performance of the system. Depending on the dataset used, (COCO in this model) some kind of angles can decrease model accuracy and also increase number of occlusions with the problems this generates in detection.
Model accuracy is relevant due to the amount of false positives or negatives it can generate degrading system performance.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
- [Model Source]
- I converted the model to an Intermediate Representation with the following arguments...
- The model was insufficient for the app because...
- I tried to improve the model for the app by...
- Model 2: [Name]
- [Model Source]
- I converted the model to an Intermediate Representation with the following arguments...
- The model was insufficient for the app because...
- I tried to improve the model for the app by...

- Model 3: [Name]
- [Model Source]
- I converted the model to an Intermediate Representation with the following arguments...
- The model was insufficient for the app because...
- I tried to improve the model for the app by...
