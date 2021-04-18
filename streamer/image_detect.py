from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from mxnet import gluon, nd
import math
# im_fname = 'test.jpg'
# im_fname = "data/000000050380.jpg"
# im_fname = "data/lana_1.png"
# im_fname = "data/satria_1.png"
# im_fname = "data/persons_1.jpg"
im_fname = '/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/sit/jos1.jpg'
model_json = 'model/Ultralight-Nano-SimplePose.json'
model_params = "model/Ultralight-Nano-SimplePose.params"

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = gluon.SymbolBlock.imports(model_json, ['data'], model_params)

detector.reset_class(["person"], reuse_weights=["person"])


x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

print('Shape of pre processed image: ', x.shape)
class_IDs, scores, bounding_boxs = detector(x)

# print("Class ID: \nlength",len(class_IDs),"\n class IDs: ",class_IDs)
# print("scores :",scores[0],"\nlen scores: ",len(scores[0]))
# print("bounding box: ",bounding_boxs,"\nLen Bounding Box: ",len(bounding_boxs))
# print("bounding box: ", bounding_boxs[0][1])

pose_input, upscale_bbox = detector_to_simple_pose(
    img, class_IDs, scores, bounding_boxs)
# print("pose input: \n",pose_input)
predicted_heatmap = pose_net(pose_input)

pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
# print("len pred_coords: ",len(pred_coords),"\tkeypoints detected: ",len(pred_coords[0]),"\tpred_coords :",str(pred_coords))

print("\n\n\t\t==========================================\n\n")
i = 0
central = []
for x in scores[0]:
    
    if x > 0.5:
        coord = pred_coords[i]
        print("scores: \t",x.asscalar())
        print("\t\tbounding_box:\t", bounding_boxs[0][i])
        central_x = ((bounding_boxs[0][i][0] + bounding_boxs[0][i][2]) / 2).asscalar()
        central_y = ((bounding_boxs[0][i][1] + bounding_boxs[0][i][3]) / 2).asscalar()
        print("\t\tCENTRAL x: ", central_x, "\n\t\tCENTRAL y: ", central_y)
        cent_5_x = (coord[5][1].asscalar() - central_x) ** 2
        cent_5_y = (coord[5][0].asscalar() - central_x) ** 2
        cent_5 = math.sqrt(cent_5_x + cent_5_y)
        print("\t\tbahu ke centroid: ",cent_5)
    elif x == -1:
        break
    i += 1


for coord in pred_coords:
    i = 0
    for x in coord:
        print(i, ": ", coord[i, 1].asscalar())
        i += 1


# posisi = []
# for coord in pred_coords:
#     y_coord = coord[:, 1]

#     y_high = max(y_coord)
#     y_low = min(y_coord)
#     selisih = y_high - y_low
#     if abs(coord[13][1] - coord[5][1]) >= (selisih * 0.6):
#         posisi.append("berdiri")
#     elif abs(coord[13][1] - coord[5][1]) < (selisih * 0.6):
#         posisi.append("jongkok")
    # print("selisih: ",selisih,"\nhigh: ",y_high,"\ny_low",y_low)
# if (len(posisi)) > 0:
#     print("banyak orang yg terdetect: ", len(posisi))
#     print("posisi: ", posisi)


# print("len confidence: ",len(confidence[0]),"\nconfidence : \n",confidence)
# print("\nbounding_boxs:",bounding_boxs)
# print("\nScores:",scores)


ax = utils.viz.plot_keypoints(img, pred_coords, confidence, class_IDs,
                              bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)
plt.show()
