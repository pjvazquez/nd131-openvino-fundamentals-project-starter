# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import math
import logging as log
from openvino.inference_engine import IENetwork, IECore

FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.iecore = None
        self.ienetwork = None
        self.input_blob = None
        self.output_blob = None
        self.ienetwork_exec = None
        self.input_image_shape = None
        

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        self.iecore = IECore()

        ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.iecore.add_extension(cpu_extension, "CPU")    
        
        model_xml_file = model
        model_weights_file = os.path.splitext(model_xml_file)[0]+".bin"
        
        self.ienetwork = IENetwork(model=model_xml_file, weights = model_weights_file)
        
        ### TODO: Check for supported layers ###
        network_supported_layers = self.iecore.query_network(network=self.ienetwork, device_name="CPU")
        
        not_supported_layers = []
        for layer in self.ienetwork.layers.keys():
            if layer not in network_supported_layers:
                not_supported_layers.append(layer)
        if len(not_supported_layers)>0:
            log.debug("Not supported layers in model: ".format(not_supported_layers))
            exit(1)
        
        self.ienetwork_exec = self.iecore.load_network(self.ienetwork, device)
        self.input_blob = next(iter(self.ienetwork.inputs))
        self.output_blob = next(iter(self.ienetwork.outputs))
        
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.iecore

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.ienetwork.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        self.input_image_shape = frame.shape
        self.ienetwork_exec.start_async(
            request_id=request_id,
            inputs={self.input_blob: frame})
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
  
        return

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.ienetwork_exec.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id, out_shape, prob_threshold):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        output = self.ienetwork_exec.requests[request_id].outputs
        objects = []
        for layer_name, out_blob in output.items():
            out_blob = out_blob.reshape(self.ienetwork.layers[self.ienetwork.layers[layer_name].parents[0]].shape)
            layer_params = YOLO(self.ienetwork.layers[layer_name].params, out_blob.shape[2])
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, self.input_image_shape[2:],out_shape[:-1],layer_params, prob_threshold )

        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['class_id'] == 0:
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if intersection_over_union(objects[i], objects[j]) > 0.2:
                        objects[j]['confidence'] = 0
        # logger.debug("OBJECTS: {}".format(objects))
        return objects

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

class YOLO:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = int(param['num'])
        self.coords = int(param['coords'])
        self.classes = int(param['classes'])
        self.anchors = [float(a) for a in param['anchors'].split(',')]

        mask = [int(idx) for idx in param['mask'].split(',')]
        self.num = len(mask)

        mask_anchors = []
        for idx in mask:
            mask_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        self.anchors = mask_anchors
        self.side = side
        self.isYoloV3 = 'mask' in param


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = math.exp(predictions[box_index + 2 * side_square])
                h_exp = math.exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects
