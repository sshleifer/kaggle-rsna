dataset = dataset_val

# set color for class
C1 = (.2, .2, .9)
C2 = (.941, .204, .204)


def get_colors_for_class_ids(class_ids, color):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append(color)
    return colors


resize_factor = 1


def show_image_id(i, image_id, r, ax=None, display_thresh=0.9):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image(
        dataset_val, inference_config, image_id)

    for i, bbox in enumerate(r['rois']):
        print(bbox)
        score = r['scores'][i]
        if score < display_thresh:
            break
        x1 = int(bbox[0] * resize_factor)
        y1 = int(bbox[1] * resize_factor)
        x2 = int(bbox[2] * resize_factor)
        y2 = int(bbox[3] * resize_factor)  # + y1
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id, C1),
                                ax=ax)


def show_image2(image_id, boxes, ax=None, display_thresh=0.9):
    resize_factor = 4
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image(dataset_val, inference_config, image_id)
    resize_factor = original_image.shape[0] / 256.
    for i, bbox in enumerate(boxes):
        print(bbox)
        score = r['scores'][i]
        if score < display_thresh:
            break
        bbox = bbox[2:]
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2] * resize_factor)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
    ax = show_img(original_image)
