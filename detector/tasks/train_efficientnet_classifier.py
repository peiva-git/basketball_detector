import os

from detector.dataset_builders import ClassificationSequenceBuilder
from detector.models import get_classification_model_callbacks
from detector.models.classification import EfficientNet

import tensorflow as tf

if __name__ == '__main__':
    builder = ClassificationSequenceBuilder('/home/ubuntu/classification_dataset/pallacanestro_trieste/', 64)
    train_sequence, val_sequence = builder.training_sequence, builder.validation_sequence

    classifier = EfficientNet(number_of_classes=2)
    classifier.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    classifier.model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=100,
        callbacks=get_classification_model_callbacks(classifier.model_name,
                                                     early_stop_patience=10,
                                                     reduce_lr_patience=5)
    )
    classifier.model.save(filepath=os.path.join('out', 'models', 'Keras_v3',
                                                classifier.model_name + '.keras'))
    classifier.model.save(filepath=os.path.join('out', 'models', 'TF', classifier.model_name),
                          save_format='tf')
    classifier.model.save(filepath=os.path.join('out', 'models', 'HDF5', classifier.model_name + '.h5'),
                          save_format='h5')

