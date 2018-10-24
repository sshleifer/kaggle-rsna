for shape in tqdm_notebook(SHAPES2):
    log_dir = Path('/home/paperspace/retinanet_shape_{}_{}/'.format(*shape))
    log_dir.mkdir(exist_ok=True)
    if max(shape) < 400:
        batch_size = 8
    else:
        batch_size = 2
    kw = dict(image_min_side=shape[0], image_max_side=shape[1], batch_size=batch_size)
    train_gen = CSVGenerator('train_path2.csv', class_map_path, **kw)
    val_gen = CSVGenerator('val_path_small_2.csv', class_map_path, **kw)
    model, training_model, prediction_model = create_models(
        backbone_retinanet=models.backbone('resnet50').retinanet,
        num_classes=train_gen.num_classes(),
        weights=wt_50_path,
        multi_gpu=0,
        freeze_backbone=False,
        config=None
    )
    callbacks = make_callbacks(log_dir, model, prediction_model)
    history = model.fit_generator(
        train_gen, steps_per_epoch=int(10000 / kw['batch_size']), epochs=10, verbose=1,
        callbacks=callbacks
    )
    hist_df = pd.DataFrame(model.history.history)
    hist_df.to_csv(log_dir / 'history.csv')
    model_path = max(sorted(log_dir.glob('*.h5')))
    print(shape, model_path)
    model, training_model, prediction_model = create_models(
        backbone_retinanet=models.backbone('resnet50').retinanet,
        num_classes=train_gen.num_classes(),
        weights=model_path,
        multi_gpu=0,
        freeze_backbone=False,
        config=None,
        nms_threshold=.01
    )
    model_slug = os.path.basename(model_path)[:-3]
    sub_path = 'sub_retnet_{}_{}_{}.csv'.format(shape[0], shape[1], model_slug)
    if os.path.exists(sub_path):
        sub_path = 'v_1{}'.format(sub_path)
        assert not os.path.exists(sub_path)
    sub_paths.append(sub_path)
    print(sub_path)
    make_submission(sub_path)

    te_preds = _get_detections(test_gen, prediction_model)
    val_preds = _get_detections(val_gen, prediction_model)

    val_det_path = 'val_dets__{}_{}_{}.pkl'.format(shape[0], shape[1], model_slug)
    te_det_path = 'te_dets__{}_{}_{}.pkl'.format(shape[0], shape[1], model_slug)
    pickle_save(te_preds, te_det_path)
    pickle_save(val_preds, val_det_path)
    model_slug = os.path.basename(model_path)[:-3]
