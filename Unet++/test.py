import os
import numpy as np
from config import result_path, test_dir2, test_dir1, weight_path
import matplotlib.image as mpig
from Unet import Nest_Net2

def write_img(pred_images, filename):

    pred = pred_images[0]
    COLORMAP = [[0, 0, 0], [255, 255, 255]]
    cm = np.array(COLORMAP).astype(np.uint8)

    pred = np.argmax(np.array(pred), axis=2)

    pred_val = cm[pred]
    mpig.imsave(os.path.join(result_path,filename.split("/")[-1]), pred_val)
    print(os.path.join("data",filename.split("/")[-1])+"finished")

model = Nest_Net2([256, 256, 6],2)

model.load_weights(weight_path+'UNET++.ckpt')

test_list_dir1 = os.listdir(test_dir1)
test_list_dir1.sort()
test_list_dir2 = os.listdir(test_dir2)
test_list_dir2.sort()
test_filenames1 = [test_dir1 + filename for filename in test_list_dir1]
test_filenames2 = [test_dir2 + filename for filename in test_list_dir2]

for filename1, filename2 in zip(test_filenames1, test_filenames2):
    image1 = mpig.imread(filename1)
    image2 = mpig.imread(filename2)
    image = np.concatenate((image1, image2), axis=2)
    image = image[np.newaxis, :, :, :].astype("float32")
    out = model.predict(image)  # out的维度为[batch, h, w, n_class]
    write_img(out, filename1)
