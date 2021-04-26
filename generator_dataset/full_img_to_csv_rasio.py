import glob
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from mxnet import gluon, nd
import csv
import math

# loading model
model_json = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.json"
model_params = "/home/jos/tf_lab/Ultralight-SimplePose/model/Ultralight-Nano-SimplePose.params"
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = gluon.SymbolBlock.imports(model_json, ['data'], model_params)
detector.reset_class(["person"], reuse_weights=["person"])


def squat(savefilename):
    for i in range(0, 301):
        print("\t\t\t========\tsquat i:", i, "\t\t========\n")
        im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/2_res_image/squat/jos' + \
            str(i)+'.jpg'
        # print('Shape of pre processed image: ', x.shape)
        x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
        class_IDs, scores, bounding_boxs = detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(
            img, class_IDs, scores, bounding_boxs)
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(
            predicted_heatmap, upscale_bbox)
        k = 0
        for x in scores[0]:
            if x > 0.5:
                rasio_central_vertical = []
                coord = pred_coords[k]
                central_x = (
                    (bounding_boxs[0][k][0] + bounding_boxs[0][k][2]) / 2).asscalar()
                central_y = (
                    (bounding_boxs[0][k][1] + bounding_boxs[0][k][3]) / 2).asscalar()
                j = 0
                # vertical = bounding_boxs[0][k][3].asscalar()-bounding_boxs[0][k][1].asscalar()
                diag_x = (bounding_boxs[0][k][0].asscalar() - central_x) ** 2
                diag_y = (bounding_boxs[0][k][1].asscalar() - central_y) ** 2
                diag = math.sqrt(diag_x + diag_y)
                print("vertical : ", diag)
                for x in coord:
                    cent_x = (x[0].asscalar() - central_x) ** 2
                    cent_y = (x[1].asscalar() - central_y) ** 2
                    distance = math.sqrt(cent_x + cent_y)
                    distance = round(distance, 4)
                    rasio = distance / diag
                    rasio_central_vertical.append(round(rasio, 6))
                    j += 1
                # variabel for squat 1
                # variabel for stand 2
                # variabel for sit 3
                rasio_central_vertical.append(1)
                with open(savefilename, mode='a') as squat:
                    distance_writer = csv.writer(
                        squat, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    distance_writer.writerow(rasio_central_vertical)
            elif x == -1:
                break
                k += 1


def stand(savefilename):
    for i in range(0, 310):
        print("\t\t\t========\tstand i:", i, "\t\t========\n")
        im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/2_res_image/stand/jos' + \
            str(i)+'.jpg'
        # print('Shape of pre processed image: ', x.shape)
        x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
        class_IDs, scores, bounding_boxs = detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(
            img, class_IDs, scores, bounding_boxs)
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(
            predicted_heatmap, upscale_bbox)
        k = 0
        for x in scores[0]:
            if x > 0.5:
                rasio_central_vertical = []
                coord = pred_coords[k]
                central_x = (
                    (bounding_boxs[0][k][0] + bounding_boxs[0][k][2]) / 2).asscalar()
                central_y = (
                    (bounding_boxs[0][k][1] + bounding_boxs[0][k][3]) / 2).asscalar()
                j = 0
                # vertical = bounding_boxs[0][k][3].asscalar()-bounding_boxs[0][k][1].asscalar()
                diag_x = (bounding_boxs[0][k][0].asscalar() - central_x) ** 2
                diag_y = (bounding_boxs[0][k][1].asscalar() - central_y) ** 2
                diag = math.sqrt(diag_x + diag_y)
                print("vertical : ", diag)
                for x in coord:
                    cent_x = (x[0].asscalar() - central_x) ** 2
                    cent_y = (x[1].asscalar() - central_y) ** 2
                    distance = math.sqrt(cent_x + cent_y)
                    distance = round(distance, 4)
                    rasio = distance / diag
                    rasio_central_vertical.append(round(rasio, 6))
                    j += 1
                    # print("rasio_central_vertical: ",rasio_central_vertical)
                # variabel for squat 1
                # variabel for stand 2
                # variabel for sit 3
                rasio_central_vertical.append(2)
                with open(savefilename, mode='a') as squat:
                    distance_writer = csv.writer(
                        squat, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    distance_writer.writerow(rasio_central_vertical)
            elif x == -1:
                break
                k += 1


def sit(savefilename):
    for i in range(100, 301):
        print("\t\t\t========\tsit i:", i, "\t\t========\n")
        im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/2_res_image/sit/jos' + \
            str(i)+'.jpg'
        # print('Shape of pre processed image: ', x.shape)
        x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
        class_IDs, scores, bounding_boxs = detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(
            img, class_IDs, scores, bounding_boxs)
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(
            predicted_heatmap, upscale_bbox)
        k = 0
        for x in scores[0]:
            if x > 0.5:
                rasio_central_vertical = []
                coord = pred_coords[k]
                central_x = (
                    (bounding_boxs[0][k][0] + bounding_boxs[0][k][2]) / 2).asscalar()
                central_y = (
                    (bounding_boxs[0][k][1] + bounding_boxs[0][k][3]) / 2).asscalar()
                j = 0
                # vertical = bounding_boxs[0][k][3].asscalar()-bounding_boxs[0][k][1].asscalar()
                diag_x = (bounding_boxs[0][k][0].asscalar() - central_x) ** 2
                diag_y = (bounding_boxs[0][k][1].asscalar() - central_y) ** 2
                diag = math.sqrt(diag_x + diag_y)
                print("vertical : ", diag)
                for x in coord:
                    cent_x = (x[0].asscalar() - central_x) ** 2
                    cent_y = (x[1].asscalar() - central_y) ** 2
                    distance = math.sqrt(cent_x + cent_y)
                    distance = round(distance, 4)
                    rasio = distance / diag
                    rasio_central_vertical.append(round(rasio, 6))
                    j += 1
                # variabel for squat 1
                # variabel for stand 2
                # variabel for sit 3
                rasio_central_vertical.append(3)
                with open(savefilename, mode='a') as squat:
                    distance_writer = csv.writer(
                        squat, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    distance_writer.writerow(rasio_central_vertical)
            elif x == -1:
                break
                k += 1


if __name__ == '__main__':
    savefile = 'generator_dataset/dataset_file_5.csv'
    stand(savefile)
    squat(savefile)
    sit(savefile)
