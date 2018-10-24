# kaggle-rsna competition code

### Best Approach (Public LB: 0.199)

- train keras-retinanet with modification to allow configuration of NMS from cli
    - that change is here https://github.com/sshleifer/keras-retinanet/commit/2f9bd031edb76348a00d97d938811d188833da58
- experiment with params: 256x256 with batch_size=12, coco starting weights, resnet50
- run inference with `nms_threshold=.01`
- calc each images probability as the highest bbox probability
- only submit detections for the 360 test images with the highest image proba
- `see retinanet_inference.py`

### Utilities other stuff that didn't help:

train mask-rcnn
train keras-retinanet on 256x256
train keras-retinanet on 300x300
no NIH data
inference: `retinanet_inference.py` 
    use nms_threshold=0.01 (combine boxes if there is any overlap)
    use score_threshold=.15 for retinanet, .95 for mask-rcnn
    
Optionally:
    can also ensemble resnet101 backed mask-rcnn
    
This repo is mostly modifications to the mask-rcnn code and the utils for ensembling.
The ensembling didn't work.
 


see `thrensemble()`: never combines detections across models, just chooses which models to use
 
