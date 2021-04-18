import glob
import cv2
sit = []


for img in glob.glob("/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/res_image/sit/*.jpg"):
    n = cv2.imread(img)
    sit.append(n)
print("length: ",len(sit))
count = 0
for image in sit:
    res = cv2.resize(image, (150,200))
    cv2.imwrite('/home/jos/tf_lab/Ultralight-SimplePose/generator_dataset/fix_image/sit/sit'+str(count)+'.jpg',res)
    count+=1
    print("image ke - ",count)