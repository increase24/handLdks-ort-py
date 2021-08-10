import argparse
import cv2
import onnxruntime as rt
import numpy as np
import pandas as pd
import torch
import glob
import math
import time
import os
os.sys.path.append('.')
from utils.camera import VideoStream
from utils.anchors import generate_anchors, anchor_options_v1, anchor_options_v2
from utils.detection import _decode_boxes, _weighted_non_max_suppression


def transform_to_targetDim(image, targetDim):
    (target_w, target_h) = targetDim
    height, width  = image.shape[0], image.shape[1]
    # padding width edge
    if(width/height < target_w/target_h):
        pad_length = int((height/target_h*target_w-width)/2)
        image_padding = cv2.copyMakeBorder(image, 0, 0, pad_length, pad_length, cv2.BORDER_CONSTANT, value=0)
    # padding height edge
    else:
        pad_length = int((width/target_w*target_h-height)/2)
        image_padding = cv2.copyMakeBorder(image, pad_length, pad_length, 0, 0, cv2.BORDER_CONSTANT, value=0)
    image_resize = cv2.resize(image, (target_w, target_h), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)
    return image_padding, image_resize

str2bool = lambda x: (str(x).lower() == 'true')
def parse_args():
    parser = argparse.ArgumentParser(description='Test onnxruntime model of hand gesture recognition')
    parser.add_argument('--palmDetModelVersion', 
    default='v2',  
    help='The version of blaze palm model'
    )
    parser.add_argument('--palmDetModelPath', 
    default='./models/palm_detection_v2.onnx',  
    help='The path of palm detection model'
    )
    parser.add_argument('--handLdksModelPath', 
    default='./models/hand_landmarks_v2.onnx',  
    help='The encoder path of dynamic gesture recognition model'
    )
    parser.add_argument('--cam_idx',
    default=0,
    type=int,
    help='camera index')
    parser.add_argument('--cam_fps',
    default=10,
    type=int,
    help='camera frame rate')
    parser.add_argument('--test_input_path',
    default='',
    type=str,
    help='read local test images')
    parser.add_argument('--use_bbox_file',
    type=str2bool,
    default='False')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ''' gestures definition '''
    sess_palmDet = rt.InferenceSession(args.palmDetModelPath)
    minScoreThr = 0.7
    # generate anchors
    if args.palmDetModelVersion == "v1":
        input_nodes_palmDet = ["input"]
        output_nodes_palmDet = ["output1","output2"]
        dim_palmDet = (256, 256)
        anchors = generate_anchors(anchor_options_v1)
    elif args.palmDetModelVersion == "v2":
        input_nodes_palmDet = ["input"]
        output_nodes_palmDet = ["regressors","classificators"]
        dim_palmDet = (128, 128)
        anchors = generate_anchors(anchor_options_v2)
    anchors = np.array(anchors)
    # hand landmarks model
    sess_handLdks = rt.InferenceSession(args.handLdksModelPath)
    input_nodes_handLdks = ["input_1"]
    output_nodes_handLdks = ["Identity", "Identity_1", "Identity_2"]
    dim_handLdks = (224, 224)

    ''' camera configuration '''
    if args.cam_idx >= 0:
        #cap = cv2.VideoCapture(args.cam_idx)
        video_stream = VideoStream(video_source = 0, fps = args.cam_fps, queue_size = 1)
        video_stream.start()
    cnt = 0
    while(True):
        ret, frame = video_stream.get_image()    
        if(ret):
            #image = cv2.flip(frame, 1)
            image = frame
            image_padding, image_resize = transform_to_targetDim(image, dim_palmDet)
            showFrame = image_padding.copy()
            org_image = image_padding.copy()
            image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            # image = np.transpose(image, [2, 0, 1])
            image = image[np.newaxis, :, :, :]
            result_palmDet = sess_palmDet.run(output_nodes_palmDet, {input_nodes_palmDet[0]: image})
            rawBoxes, scores  = result_palmDet
            decodedBoxes = _decode_boxes(rawBoxes, anchors, dim_palmDet)
            detBoxes_tensor = torch.from_numpy(decodedBoxes)
            raw_score_tensor = torch.from_numpy(scores)
            raw_score_tensor = raw_score_tensor.clamp(-100.0, 100.0)
            detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
            mask = detection_scores >= minScoreThr
            for i in range(detBoxes_tensor.shape[0]):
                boxes = detBoxes_tensor[i][mask[i]]
                scores = detection_scores[i][mask[i]].unsqueeze(-1)
                outDet = _weighted_non_max_suppression(torch.cat([boxes, scores], -1), 0.3)
                if(len(outDet)>0):
                    for det_t in outDet:
                        ymin = det_t[0] * showFrame.shape[0] 
                        xmin = det_t[1] * showFrame.shape[1] 
                        ymax = det_t[2] * showFrame.shape[0] 
                        xmax = det_t[3] * showFrame.shape[1] 
                        # ----- palm region -> hand region ------ #
                        palmUpId = 2 
                        palmDownId = 0
                        handUp = (det_t[4 + palmUpId * 2],  det_t[4 + palmUpId * 2 + 1])
                        handDown = (det_t[4 + palmDownId * 2],  det_t[4 + palmDownId * 2 + 1])
                        xscale = xmax - xmin
                        yscale = ymax - ymin
                        shift_x = 0
                        shift_y = 0.5
                        palm_box_scale = 2.6
                        angleRad = math.atan2(handDown[0]-handUp[0], handDown[1] - handUp[1])
                        x_center = xmin + xscale * (0.5 - shift_y * math.sin(angleRad) + shift_x * math.cos(angleRad))
                        y_center = ymin + yscale * (0.5 - shift_y * math.cos(angleRad) - shift_x * math.sin(angleRad))
                        x_leftTop = max(int(x_center - xscale * palm_box_scale/2) ,0)
                        y_leftTop = max(int(y_center - yscale * palm_box_scale/2) ,0)
                        x_rightBottom = min(int(x_center + xscale * palm_box_scale/2), showFrame.shape[0])
                        y_rightBottom = min(int(y_center + yscale * palm_box_scale/2), showFrame.shape[1])
                        cv2.rectangle(showFrame, (x_leftTop, y_leftTop), (x_rightBottom, y_rightBottom), (255, 0, 0), 2)

                        img_handRegion = org_image[y_leftTop:y_rightBottom, x_leftTop:x_rightBottom]
                        img_handRegion_pad, img_handRegion_resize = transform_to_targetDim(img_handRegion, dim_handLdks)
                        img_handRegion_resize = img_handRegion_resize.astype(np.float32) / 255.0
                        #img_handRegion = np.transpose(img_handRegion, [2, 0, 1])
                        img_input = img_handRegion_resize[np.newaxis, :, :, :]
                        result_handLdks = sess_handLdks.run(output_nodes_handLdks, {input_nodes_handLdks[0]: img_input})
                        hand_kpts, hand_score, handness = result_handLdks

                        # visualize the keypoints
                        for kdx in range(hand_kpts.shape[1]//3):
                            x_kpt = hand_kpts[0,kdx*3]/dim_handLdks[0]*img_handRegion_pad.shape[1]
                            y_kpt = hand_kpts[0,kdx*3+1]/dim_handLdks[1]*img_handRegion_pad.shape[0]
                            if(img_handRegion.shape[0]>img_handRegion.shape[1]): # h>w
                                x_kpt = x_kpt - (img_handRegion.shape[0] - img_handRegion.shape[1])
                            elif(img_handRegion.shape[1]>img_handRegion.shape[0]): # w>h
                                y_kpt = y_kpt - (img_handRegion.shape[1] - img_handRegion.shape[0])
                            x_kpt, y_kpt = int(x_kpt) + x_leftTop, int(y_kpt) + y_leftTop
                            cv2.circle(showFrame, (x_kpt, y_kpt), 1, (255,255,255), 2)



            cv2.imshow("hand landmarks", showFrame)
            if(cv2.waitKey(5) == 27):
                video_stream.stop()
                break


if __name__ == "__main__":
    main()