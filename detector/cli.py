import argparse
import os.path

import tensorflow as tf

import detector.models
from .dataset_builders import ClassificationDatasetBuilder
from .models.classification import SimpleClassifier


def train_command(debug_enabled: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='The model to train',
        choices=['simple-classifier', 'resnet'],
        type=str,
        required=True
    )
    parser.add_argument(
        '--dataset-dir',
        help='Dataset root directory',
        type=str,
        required=True
    )
    parser.add_argument(
        '--epochs',
        help='Number of training epochs',
        type=int,
        required=True
    )
    args = parser.parse_args()

    builder = ClassificationDatasetBuilder(args['dataset-dir'])
    builder.configure_datasets_for_performance()
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    if debug_enabled:
        train_dataset = train_dataset.take(int(len(train_dataset) * 0.05))
        validation_dataset = validation_dataset.take(int(len(validation_dataset) * 0.05))

    if args.model == 'simple-classifier':
        model = SimpleClassifier(model_name=args.model).model
    else:
        model = SimpleClassifier(model_name='resnet-classifier').model

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.CategoricalAccuracy()]
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=detector.models.get_classification_model_callbacks(
            early_stop_patience=int(0.3 * args.epochs),
            reduce_lr_patience=int(0.2 * args.epochs)
        )
    )
    model.save(filepath=os.path.join('out', 'models', 'TF', args.model), save_format='tf')
    model.save(filepath=os.path.join('out', 'models', 'HDF5', args.model), save_format='h5')
