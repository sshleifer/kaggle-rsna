import numpy as np
from tqdm import tqdm_notebook
import pydicom


def run_inference_val(model, dataset, config, image_ids=None):
    if image_ids is None:
        image_ids = dataset.image_ids
    detections = []
    for image_id in tqdm_notebook(image_ids):
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id,
        )
        results = model.detect([image], verbose=0)
        r = results[0]
        # r.pop('masks')  # makes inspection annoying cause adds 256 x 256 foreach
        detections.append(r)
    return detections


def run_inference_test(model, dicom_paths, config=None):
    if config is None:
        min_dim, min_scale, max_dim, resize_mode = (256, 0, 256, 'square')
    else:
        min_dim, min_scale, max_dim, resize_mode = (
            config.IMAGE_MIN_DIM, config.IMAGE_MIN_SCALE, config.IMAGE_MAX_DIM, config.IMAGE_RESIZE_MODE
        )
    detections = {}
    for path in tqdm_notebook(dicom_paths):
        ds = pydicom.read_file(path)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        image, window, scale, padding, crop = utils.resize_image(
            image, min_dim=min_dim, min_scale=min_scale, max_dim=max_dim, mode=resize_mode
        )
        assert image.shape == (min_dim, min_dim, 3), image.shape
        results = model.detect([image])
        r = results[0]
        detections[path] = r
    return detections
