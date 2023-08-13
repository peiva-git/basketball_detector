import os
import tensorflow as tf

from detector.models.classification import SimpleClassifier
from detector.dataset_builders import  ClassificationDatasetBuilder


def train_classifier_on_reduced_dataset(model: tf.keras.models.Model):
    print('Reading files from disk...')
    builder = ClassificationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste/',
                                           reduce_percentage=0.95)
    builder.configure_datasets_for_performance()
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

#    print('Reducing training and validation datasets to 5% of the original size...')
#   train_dataset = train_dataset.take(int(tf.data.experimental.cardinality(train_dataset).numpy() * 0.05))
#    validation_dataset = validation_dataset\
#        .take(int(tf.data.experimental.cardinality(validation_dataset).numpy() * 0.05))
#    print(tf.data.experimental.cardinality(train_dataset).numpy(), 'images in training dataset')
#    print(tf.data.experimental.cardinality(validation_dataset).numpy(), 'images in validation dataset')

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=model.get_model_callbacks(model.model_name, early_stop_patience=3, reduce_lr_patience=2)
    )
    model.save(filepath=os.path.join('out', '../models', 'TF', model.model_name), save_format='tf')
    model.save(filepath=os.path.join('out', '../models', 'HDF5', model.model_name + '.h5'), save_format='h5')


if __name__ == '__main__':
    train_classifier_on_reduced_dataset(SimpleClassifier().detector)
