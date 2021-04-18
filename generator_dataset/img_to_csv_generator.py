import glob
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from mxnet import gluon, nd
import mxnet as mx
import csv
import math
import cv2
# loading model

model_json = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.json"
model_params = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.params"
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = gluon.SymbolBlock.imports(model_json, ['data'], model_params)
detector.reset_class(["person"], reuse_weights=["person"])


for i in range(0, 57):
    print("\t\t\t  ========\t  i:", i,"\t========\n")


    # SQUAT range(0, 56):
    # im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/squat/jos'+str(i)+'.jpg'
 
    # STAND range(2, 57):
    # im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/stand/jos'+str(i)+'.jpg'

    # SIT range(0, 57):
    im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/sit/jos'+str(i)+'.jpg'

    x, img = data.transforms.presets.ssd.load_test(im_fname, short=512) # print('Shape of pre processed image: ', x.shape)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    k = 0
    for x in scores[0]:
        if x > 0.5:
            distance_from_central = []
            coord = pred_coords[k]
            central_x = ((bounding_boxs[0][k][0] + bounding_boxs[0][k][2]) / 2).asscalar()
            central_y = ((bounding_boxs[0][k][1] + bounding_boxs[0][k][3]) / 2).asscalar()
            j = 0
            for x in coord:
                # print(j, ": ", x)
                cent_x = (x[0].asscalar() - central_x) ** 2
                cent_y = (x[1].asscalar() - central_y) ** 2
                distance = math.sqrt(cent_x + cent_y)
                distance = round(distance, 4)
                distance_from_central.append(distance)
                # print(j, ": ", coord[j, 1].asscalar())
                j += 1
            # variabel for squat 1
            # variabel for stand 2
            # variabel for sit 3
            distance_from_central.append(3)
            with open('generator_dataset/dataset_file.csv', mode='a') as squat:
                distance_writer = csv.writer(
                    squat, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                distance_writer.writerow(distance_from_central)
        elif x == -1:
            break
            k += 1