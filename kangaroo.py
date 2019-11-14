import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mrcnn.config import Config
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap

import warnings
warnings.filterwarnings("ignore")

Server = True
Server = False

Train = True
Train = False

class KangarooDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):

        self.add_class("dataset", 1, "kangaroo")

        images = dataset_dir + '/images/'
        annots = dataset_dir + '/annots/'

        for filename in sorted(os.listdir(images)):
            img_id = filename[:-4]

            if img_id in ['00090']:
                continue

            if is_train and int(img_id) >= 150:
                continue

            if not is_train and int(img_id) < 150:
                continue

            img_path = images + filename
            ann_path = annots + img_id + '.xml'

            self.add_image('dataset', image_id=img_id, path=img_path, annotation=ann_path)

    def extract_bboxes(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()
        
        # extract each bounding box
        
        boxes = []
        
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        # extract image dimensions
        
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
    
        return boxes, width, height

    def load_mask(self, img_id):

        info = self.image_info[img_id]
        # define bbox coords location
        path = info['annotation']
        # load XML for file
        boxes, w, h = self.extract_bboxes(path)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []

        for i in range(len(boxes)):
            box = boxes[i]
            row_min, row_max = box[1], box[3]
            col_min, col_max = box[0], box[2]

            masks[row_min:row_max, col_min:col_max, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))

        return masks, np.asarray(class_ids, dtype='uint8')

    # load an image reference
    def image_reference(self, img_id):
        info = self.image_info[img_id]
        return info['path']

dataset_dir = os.getcwd() + '/kangaroo'

# training set

train_set = KangarooDataset()
train_set.load_dataset(dataset_dir)
train_set.prepare()

# testing set

test_set = KangarooDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()

image_id = 21
# image = train_set.load_image(image_id)
# mask, class_ids = train_set.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image, bbox, mask, class_ids, train_set.class_names)

class KangarooConfig(Config):
    NAME = 'kangaroo_config'
    # State number of classes inc. background
    NUM_CLASSES = 1 + 1
    LEARNING_RATE = 0.01
    STEPS_PER_EPOCH = 131

# prepare config

config = KangarooConfig()
config.display()
#define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)

# load weights (mscoco)
if Train:

    model.load_weights('mask_rcnn_coco.h5',
                        by_name=True,
                        exclude=["mrcnn_class_logits",
                                "mrcnn_bbox_fc",
                                "mrcnn_bbox",
                                "mrcnn_mask"])
    
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=7, layers='heads')

class PredictionConfig(Config):
    NAME = 'kangaroo_config'
    # State number of classes inc. background
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if Server:

    weights = '/home/ubuntu/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
else:
    weights = '/home/david/Projects/strath/kangaroo/models/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'

tst_weights = '/home/david/Projects/strath/data/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
# create config
cfg = PredictionConfig()

if not Train:

    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    
    model.load_weights(weights, by_name=True)

    def evaluate_model(dataset, model, cfg):
        APs = list()
        for image_id in dataset.image_ids:
            # load image, bounding boxes and masks for the image id
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = np.expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=1)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            # store
            APs.append(AP)
    
        # calculate the mean AP across all images
        mAP = np.mean(APs)
    
        print('Mean Average Precision: {}'.format(mAP))
        return mAP

    print('predicting on train set...')
    # # # evaluate model on training dataset
    # train_mAP = evaluate_model(train_set, model, cfg)
    # print("Train mAP: %.3f" % train_mAP)
    
    # test_mAP = evaluate_model(test_set, model, cfg)
    # print("Train mAP: %.3f" % test_mAP)

n_image = 2

for i in range(n_image):
    image = test_set.load_image(i)
    mask, _ = test_set.load_mask(i)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)
    # make prediction
    print('detecting...')
    yhat = model.detect(sample, verbose=0)
    # stage subplot
    plt.subplot(n_image, 2, i*2+1)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Actual')
    # plot masks
    for j in range(mask.shape[2]):
        plt.imshow(mask[:,:,j], cmap='Blues', alpha=0.3)

    plt.subplot(n_image, 2, i*2+2)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Predicted')
    ax = plt.gca()
    # plot predicted masks
    yhat = yhat[0]
    for box in yhat['rois']:
        # get coordinates
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)

plt.show()