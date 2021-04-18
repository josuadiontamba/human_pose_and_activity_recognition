from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_keypoints #cv_plot_image,  

import sys
sys.path.append('/home/jos/tf_lab/Ultralight-SimplePose/streamer')
from plot_image import cv_plot_image

def keypoint_detection(img, detector, pose_net, ctx):
    x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs, output_shape=(256,192), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        print("len pred_coords: ",len(pred_coords),"\npred_coords \n",str(pred_coords))
        # print("\nlen confidence: ",len(confidence),"confidence: ",confidence)
        scale = 1.0 * img.shape[0] / scaled_img.shape[0]
        img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.3, scale=scale)
    return img


if __name__ == '__main__':

    json_path = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.json"
    params_path = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.params"

    ctx = mx.gpu(0)
    detector_name = "yolo3_mobilenet1.0_coco"
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
    net = gluon.SymbolBlock.imports(json_path, ['data'], params_path)
    # net = gluon.SymbolBlock.imports(json_path, ['data'], params_path)
    net.collect_params().reset_ctx(ctx=ctx)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
    url = "rtsp://admin:Awasjosco21@192.168.1.7:554/onvif1"
    cap = cv2.VideoCapture(url)    

    time.sleep(1)
    num = 0
    while True:
        start = time.time()
        ret,frame = cap.read()
        # print("dtype: ",frame.dtype,"\tshape: ",frame.shape)
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_for_key = mx.nd.array(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).astype('uint8')
        
        # # print("dtype: ",frame.dtype,"\tshape: ",frame.shape)
        cv_plot_image(frame2,canvas_name='orig', start=start)
        img = keypoint_detection(frame_for_key, detector, net, ctx=ctx)
        cv_plot_image(img,canvas_name='keypoint', start=start)

        if (cv2.waitKey(1) & 0xFF == ord('k')):
            cv2.imwrite('/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/sit/jos'+str(num)+'.jpg',frame)
            num = num + 1
        if (cv2.waitKey(1) & 0xFF == ord('l')):
            break
    cap.release()
    cv2.destroyAllWindows()