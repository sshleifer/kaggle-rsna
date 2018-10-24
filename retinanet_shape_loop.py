from keras_retinanet.callbacks import *
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models import load_model
from keras_retinanet.bin.train import create_models
from keras_retinanet.models.retinanet import retinanet_bbox
import keras

wt_50_path = '/home/paperspace/keras-retinanet/snapshots/resnet50_coco_best_v2.1.0.h5'







def make_callbacks(log_dir, model, prediction_model, save_best_only=True, backbone='resnet50'):
    # needs model and prediction_model
    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    )

    evaluation = Evaluate(val_gen, save_path='logs', max_detections=4)
    evaluation = RedirectModel(evaluation, prediction_model)
    checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    log_dir,
                    '{backbone}_{{epoch:02d}}.h5'.format(
                        backbone=backbone, dataset_type='rsna')
                ),
                verbose=1,
                save_best_only=save_best_only,
                monitor="mAP", #
                mode="max",
            )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='"mAP"', min_delta=0, patience=5, verbose=1,
        mode='max', restore_best_weights=True,
    )
    checkpoint = RedirectModel(checkpoint, model)
    return [
        lr_reduce,
        early_stopping,
        evaluation,
        checkpoint,
    ]

BACKBONE= 'resnet50'
from pathlib import Path
for shape in SHAPES:
    log_dir = Path('/home/paperspace/retinanet_shape_{}_{}/'.format(*shape))
    log_dir.mkdir(exist_ok=True)
    kw = dict(image_min_side=shape[0], image_max_side=shape[1], batch_size=4)
    train_gen = CSVGenerator('train_path2.csv', class_map_path, **kw)
    val_gen = CSVGenerator('val_path_small_2.csv', class_map_path, **kw)
    model, training_model, prediction_model = create_models(
        backbone_retinanet= models.backbone(BACKBONE).retinanet,
        num_classes=train_gen.num_classes(),
        weights=wt_50_path,
        multi_gpu=0,
        freeze_backbone=True,
        config=None
    )
    callbacks = make_callbacks(log_dir, model, prediction_model)
    history = model.fit_generator(
        train_gen, steps_per_epoch=10000 / 4, epochs=40, verbose=1, callbacks=callbacks
    )
