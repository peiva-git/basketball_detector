import argparse
import os.path

import tensorflow as tf

from .dataset_builder import DatasetBuilder
from .models import SimpleClassifier, ResNetClassifier, get_model_callbacks


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

    builder = DatasetBuilder(args['dataset-dir'])
    builder.configure_datasets_for_performance()
    train_dataset, validation_dataset = builder.build()

    if debug_enabled:
        train_dataset = train_dataset.take(int(len(train_dataset) * 0.05))
        validation_dataset = validation_dataset.take(int(len(validation_dataset) * 0.05))

    if args.model == 'simple-classifier':
        model = SimpleClassifier()
    else:
        model = ResNetClassifier()
    model.compile(
        optimizer='adamw',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=get_model_callbacks(args.model, int(0.3 * args.epochs), int(0.2 * args.epochs))
    )
    model.save(filepath=os.path.join('out', 'models', args.model), save_format='tf')
