import os

from detector.dataset_builders import SegmentationDatasetBuilder
from detector.models import get_segmentation_model_callbacks

import tensorflow as tf

from detector.models.segmentation import MobileNetUNet

if __name__ == '__main__':
    builder = SegmentationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_segmentation/pallacanestro_trieste/')
    builder.configure_datasets_for_performance(shuffle_buffer_size=100, input_batch_size=10)
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    segmenter = MobileNetUNet(input_shape=(224, 224, 3), output_channels=2)
    segmenter.model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(momentum=0.3, learning_rate=0.01),
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    segmenter.model.summary()
    segmenter.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=get_segmentation_model_callbacks(segmenter.model_name, early_stop_patience=10, reduce_lr_patience=5)
    )
    segmenter.model.save(filepath=os.path.join('out', 'models', 'TF', segmenter.model_name), save_format='tf')
    segmenter.model.save(filepath=os.path.join('out', 'models', 'HDF5', segmenter.model_name + '.h5'), save_format='h5')
    segmenter.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', segmenter.model_name + '.keras'))
