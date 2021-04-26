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
from plot_image import cv_plot_image

import pickle


def keypoint_detection(img, detector, pose_net, mymodel, ctx):
    x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs, output_shape=(256,192), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        # print("len pred_coords: ",len(pred_coords),"\npred_coords \n",str(pred_coords))
        # print("\nlen confidence: ",len(confidence),"confidence: ",confidence)
        
        coord = pred_coords[0]
        central_x = ((bounding_boxs[0][0][0] + bounding_boxs[0][0][2]) / 2).asscalar()
        central_y = ((bounding_boxs[0][0][1] + bounding_boxs[0][0][3]) / 2).asscalar()
        distance_from_central = []
        for x in coord:
            cent_x = (x[0].asscalar() - central_x) ** 2
            cent_y = (x[1].asscalar() - central_y) ** 2
            distance = math.sqrt(cent_x + cent_y)
            distance = round(distance, 4)
            distance_from_central.append(distance)
        # print('type: ',type(distance_from_central),', len: ',len(distance_from_central),'distance_from_central: ',distance_from_central)
        pred = [] 
        for x in range(0,17):
            if (x == 5 or x == 6 or x == 11 or x == 12 or x == 13 or x == 14 or x == 15 or x == 16):
                pred.append(distance_from_central[x])
        # print('pred: ',pred)
        result = mymodel.predict([pred])
        print("result: ",result)
        pose = "?"
        if result[0] == 1.0: 
            # print("squat")
            pose = "squat"
        elif result[0] == 2.0: 
            # print("stand")
            pose = "stand"
        elif result[0] == 3.0: 
            # print("sit")
            pose = "sit"
        # print(result)

        scale = 1.0 * img.shape[0] / scaled_img.shape[0]
        # print("class ids type: ",type(class_IDs))
        # print("score type: ",type(scores))
        img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores, pose ,box_thresh=0.5, keypoint_thresh=0.3, scale=scale)
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

    # pose_path = "/home/jos/tf_lab/Ultralight-SimplePose/trainer/finalized_model.sav"
    pose_path = '/home/jos/tf_lab/Ultralight-SimplePose/trainer/AB_model_2.sav'
    mymodel = pickle.load(open(pose_path, 'rb'))


    time.sleep(1)
    num = 0
    while True:
        start = time.time()
        ret,frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).astype('uint8')
        # img = keypoint_detection(img, detector, pose_net, ctx, mymodel):
        img = keypoint_detection(frame, detector, net, mymodel, ctx=ctx)
        
        cv_plot_image(img, start=start)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()