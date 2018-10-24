# kaggle-rsna

Approach:

train mask-rcnn
train keras-retinanet on 256x256
train keras-retinanet on 300x300
no NIH data
inference: 
    use nms_threshold=0.01 (combine boxes if there is any overlap)
    use score_threshold=.15 for retinanet, .95 for mask-rcnn
    
Optionally:
    can also ensemble resnet101 backed mask-rcnn
    
This repo is mostly modifications to the mask-rcnn code and the utils for ensembling.
 


see `thrensemble()`: never combines detections across models, just chooses which models to use
 
