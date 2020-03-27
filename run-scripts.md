python3 object_detection_demo_yolov3_async.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml -d CPU


python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


python main.py -i CAM -m tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 640x480 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm