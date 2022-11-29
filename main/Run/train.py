import os
import tensorflow as tf

from utils.mainutils import makedirs


def train(models, args=None):
    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create the generators
    train_generator, validation_generator = create_generators(args)

    model = models['train_model']
    prediction_model = models['prediction_model']

    # create the model
    if args.snapshot:
        print('Loading model, this may take a second...')
        model.load_weights(args.snapshot, by_name=True)

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-5),
    #               loss={'centernet_loss': lambda y_true, y_pred: y_pred})

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )


def create_generators(args):
    common_args = {
        'batch_size': args.batch_size,
        'input_size': args.input_size,
    }

    from generators.circle import CircleGenerator
    train_generator = CircleGenerator(
        args.dataset_path,
        args.train_json,
        misc_effect=None,
        visual_effect=None,
        group_method='random',
        **common_args
    )

    validation_generator = CircleGenerator(
        args.dataset_path,
        args.val_json,
        shuffle_groups=False,
        **common_args
    )

    return train_generator, validation_generator


def create_callbacks(training_model, prediction_model, validation_generator, args):
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from evaluation.circle import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{model}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5'.format(model=args.model_name)
            ),
            verbose=0,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=5e-7
    ))

    return callbacks