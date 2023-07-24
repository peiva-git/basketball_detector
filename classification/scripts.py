import os
import tensorflow as tf

from classification import SimpleClassifier, DatasetBuilder, get_model_callbacks


def train_classifier_on_reduced_dataset(model: tf.keras.models.Model, model_name: str):
    print('Reading files from disk...')
    builder = DatasetBuilder('/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste/')
    builder.configure_datasets_for_performance()
    train_dataset, validation_dataset = builder.build()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print('Reducing training and validation datasets to 5% of the original size...')
    train_dataset = train_dataset.take(int(len(train_dataset) * 0.05))
    validation_dataset = validation_dataset.take(int(len(validation_dataset) * 0.05))
    print(tf.data.experimental.cardinality(train_dataset).numpy(), 'images in training dataset')
    print(tf.data.experimental.cardinality(validation_dataset).numpy(), 'images in validation dataset')

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=get_model_callbacks(model_name, early_stop_patience=3, reduce_lr_patience=2)
    )
    model.save(filepath=os.path.join('out', 'models', model_name), save_format='tf')
    model.save(filepath=os.path.join('out', 'models', model_name), save_format='h5')


if __name__ == '__main__':
    train_classifier_on_reduced_dataset(SimpleClassifier(), 'simple-classifier')
