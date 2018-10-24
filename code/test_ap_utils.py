import numpy as np

from code.ap_utils import average_precision_image, box_mask_coords


def test_average_precision_image(plotting=False):
    confidences = [0.3, 0.9]
    predicted_boxes = [[20, 20, 80, 90], [110, 110, 160, 180]]
    target_boxes = [[25, 25, 85, 95], [100, 100, 150, 170], [200, 200, 230, 250]]
    val = average_precision_image(predicted_boxes, confidences, target_boxes)
    assert round(val, 3) == .375, val
    assert average_precision_image(predicted_boxes, confidences, []) == 0
    assert average_precision_image([], [], target_boxes) == 0
    assert np.isnan(average_precision_image([], [], []))
    predicted_boxes.append([110, 110, 160, 180])
    confidences.append(.2)
    val = average_precision_image(predicted_boxes, confidences, target_boxes)
    assert round(val, 3) == .375, val

    if plotting:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        for box_p in predicted_boxes:
            plt.imshow(box_mask_coords(box_p, shape=256), cmap=mpl.cm.Reds, alpha=0.3)
        for box_t in target_boxes:
            plt.imshow(box_mask_coords(box_t, shape=256), cmap=mpl.cm.Greens, alpha=0.3)
