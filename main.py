# -*- coding: utf-8 -*-

"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from collections import deque
from csv import DictWriter

FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.ERROR)
#logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
MODEL_PATH = "/mnt/DATA/Python_Projects/nd131-openvino-fundamentals-project-starter/tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml"
VIDEO_PATH = "resources/Pedestrian_Detect_2_1_1.mp4"


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False, type=str,
                        default=MODEL_PATH,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        default=VIDEO_PATH,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=CPU_EXTENSION,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network = Network()
    infer_network_vals = infer_network.load_model(model=args.model,
                                                  device=args.device,
                                                  cpu_extension=args.cpu_extension)
    log.debug(infer_network_vals)
    input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    if args.input =='CAM':
        input_stream = 0
        single_image = False
    elif args.input[-4:] in [".jpg", ".bmp"]:
        single_image = True
        input_stream = args.input
    else:
        single_image=False
        input_stream = args.input
        assert os.path.isfile(input_stream)
        
        
    capture = cv2.VideoCapture(input_stream)
    capture.open(input_stream)
    if not capture.isOpened():
        log.error("Unable to open video source")

    logger.debug( "W+H: " + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + "-" + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    t0=0
    infer_time=0
    t1=0
    process_time=0
    request_id = 0
    total_count = 0
    previous_count = 0
    num_persons_in = 0
    current_count = 0
    stay_time = 0
    max_stay_time = 0
    mean_stay_time = 0

    track_threshold = 0.1
    max_len=30

    # this list is to transform values in an excel file
    data_list = []

    # queue to accumulate last "max_len" number of detections
    track = deque(maxlen=max_len)

    ### TODO: Loop until stream is over ###
    while capture.isOpened():
        data_element = {}
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        logger.debug("size: ".format(input_shape) )
        resized_frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
        transposed_resized_frame = resized_frame.transpose((2,0,1))
        resh_transposed_resized_frame = transposed_resized_frame.reshape(input_shape)

        ### TODO: Start asynchronous inference for specified request ###
        t0=time.time()
        infer_network.exec_net(request_id, resh_transposed_resized_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id)== 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id, frame.shape, prob_threshold)
            t1 = time.time()
            infer_time = t1 - t0
            ### TODO: Extract any desired stats from the results ###
            current_count, bb_frame = count_persons(result,frame)
            process_time = time.time() - t1
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            
            # append number of detections to "track" queue
            track.append(current_count)
            # proportion of frames with a positive detection 
            num_tracked = 0
            if np.sum(track)/max_len > track_threshold:
                num_tracked = 1
            
            if num_tracked > previous_count:
                logger.debug("INTO IF ------------------------------------")
                start_time = time.time()
                num_persons_in = num_tracked - previous_count
                total_count += num_persons_in
                previous_count = num_tracked
                client.publish("person", json.dumps({"total":total_count}), retain=True)
                # client.publish("person", json.dumps({"count":num_tracked}), retain=True)
            
            ### Topic "person/duration": key of "duration" ###
            if num_tracked < previous_count:
                previous_count = num_tracked
                # client.publish("person", json.dumps({"count":num_tracked}), retain=True)

            if num_tracked > 0:
                stay_time += (time.time() - start_time)/10
                logger.debug("Duration: {}".format(stay_time))

            if total_count > 0:
                mean_stay_time = stay_time/total_count
                client.publish("person/duration", json.dumps({"duration": int(mean_stay_time)}))

            client.publish("person", json.dumps({"count":num_tracked}), retain=True)
            
        data_element['time'] = time.strftime("%H:%M:%S", time.localtime())
        data_element['current_count'] = current_count
        data_element['num_tracked'] = num_tracked
        data_element['num_persons_in'] = num_persons_in
        data_element['previous_count'] = previous_count
        data_element['total_count'] = total_count
        data_element['stay_time'] = stay_time
        data_element['mean_stay_time'] = mean_stay_time
        data_element['infer_time'] = infer_time
        data_element['process_time'] = process_time
        data_element['result']=result

        data_list.append(data_element)

        logger.debug("NUM TRACKED: {} - {} - PREVIOUS COUNT: {} - TOTAL COUNT: {} - STAY TIME: {}".format(num_tracked, np.sum(track), previous_count, total_count, mean_stay_time))
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            write_file(data_list)
            capture.release()
            cv2.destroyAllWindows()
            client.disconnect()
            break

        ### TODO: Send the frame to the FFMPEG server ###
        logger.debug("Image_size: {}".format(bb_frame.shape))

        sys.stdout.buffer.write(bb_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
            cv2.imwrite("output.jpg", bb_frame)

    write_file(data_list)
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()

def write_file(all_data):
    with open('./spreadsheet.csv','w') as outfile:
        writer = DictWriter(outfile,('time','current_count','num_tracked',
                'num_persons_in','previous_count',
                'total_count','stay_time',
                'mean_stay_time','infer_time',
                'process_time','result'))
        writer.writeheader()
        writer.writerows(all_data)

def count_persons(detections, image):
    num_detections = 0
    frame_with_bb = image
    if len(detections) > 0:
        frame_with_bb, num_detections = get_draw_boxes(detections, image)
    return num_detections, frame_with_bb

 
def get_draw_boxes(boxes, image):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    image_h, image_w, _ = image.shape
    num_detections = 0
    for box in boxes:
        logger.debug("box: {}".format(box))
        if box['class_id'] == 0:
            if box['confidence'] > 0:
                cv2.rectangle(image,(box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0,255,0), 1)
                num_detections +=1

       
    return image, num_detections

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
