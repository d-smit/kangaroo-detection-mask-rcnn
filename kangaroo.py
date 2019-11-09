import os
import xml.etree.ElementTree as ET
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import numpy as np
import matplotlib.pyplot as plt

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

dataset_dir = '/home/david/Projects/strath/kangaroo'

# training set

train_set = KangarooDataset()
train_set.load_dataset(dataset_dir)
train_set.prepare()

image_id = 1
image = train_set.load_image(image_id)
mask, class_ids = train_set.load_mask(image_id)
bbox = extract_bboxes(mask)

display_instances(image, bbox, mask, class_ids, train_set.class_names)



