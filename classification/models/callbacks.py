import os

import tensorflow as tf

# Simple classifier implemented following the tutorial at
# https://www.tensorflow.org/tutorials/images/classification#a_basic_keras_model


def get_model_callbacks(
        model_name: str,
        early_stop_patience: int,
        reduce_lr_patience: int) -> [tf.keras.callbacks.Callback]:
    model_dir_path = os.path.join('out', 'training-callback-results', model_name)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir_path, 'checkpoint'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(model_dir_path, 'backup')),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=early_stop_patience,
            start_from_epoch=2
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir_path, 'tensorboard-logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=reduce_lr_patience,
            min_lr=0.001
        ),
        tf.keras.callbacks.ProgbarLogger()
    ]
