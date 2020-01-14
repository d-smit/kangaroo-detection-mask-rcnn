# Kangaroo Detection

This is a work-in-progress object detection project for detection of kangaroos. The dataset is made up of 183 kangaroo images 
with bounding box annotations, and can be found [here](https://github.com/experiencor/kangaroo). 

This project uses the Mask R-CNN detection and segmentation model. As the dataset does not provide image masks, we will only be using the
detection capabilities of the model. 

## Requirements

The Mask R-CNN library is needed to use this project.  

```bash
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
python setup.py install
```

## Training 

The training of the model and subsequent evaluation methods can both be found in ```kangaroo.py```. Currently, we are using a pre-trained 
Mask R-CNN model trained on the large-scale object detection dataset [MS COCO](http://cocodataset.org/#home). The model then trains on 
150 of the 183 kangaroo images, and is evaluated on 33. 

The model is being trained on an AWS EC2 instance powered by NVIDIA K80 GPU processors. The best model has been trained for 2 epochs with a
training time of 8 minutes per epoch. 

## Initial Results

For object detection, a commonly used metric is the Mean Average Precision (mAP). A bounding box prediction perfectly aligned over the annotated box gives an mAP of 1.0. An mAP above 0.50 is considered a positive detection. The model is currently reaching an mAP of 0.73 over the training set. 

![](/current_predictions.png)

We can see detections made for two images from the test set above. Whilst detecting the location of each kangaroo well, the model makes multiple detections over each target. Further training is required. 
