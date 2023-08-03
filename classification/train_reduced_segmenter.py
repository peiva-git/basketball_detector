import os

import keras.callbacks, keras.losses

from classification.dataset_builders import SegmentationDatasetBuilder
from classification.models.models import Segmentation

builder = SegmentationDatasetBuilder(
    data_directory='/mnt/SharedData2/tesi/dataset/testing-dataset/frames/',
    masks_directory='/mnt/SharedData2/tesi/dataset/testing-dataset/masks/'
)
builder.configure_datasets_for_performance(shuffle_buffer_size=1000, input_batch_size=10)
train_batches = builder.train_dataset
validation_batches = builder.validation_dataset

model = Segmentation(image_size=(512, 1024)).model
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint("basketball_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 10
model.fit(train_batches, epochs=epochs, validation_data=validation_batches, callbacks=callbacks)
model.save(filepath=os.path.join('out', 'models', 'TF', 'segmenter'), save_format='tf')
model.save(filepath=os.path.join('out', 'models', 'HDF5', 'segmenter' + '.h5'), save_format='h5')
