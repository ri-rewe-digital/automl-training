import argparse
import logging
import sys

import tensorflow as tf

from training.dataset import create_iterator, export_dataset, load_data_array, split_dataset
from training.model import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--bucket', type=str, required=True)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
FLAGS = parser.parse_args()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    logging.info('Creating dataset iterators...')
    dataset_csv = export_dataset(project=FLAGS.project, dataset=FLAGS.dataset, bucket=FLAGS.bucket)
    df, labels = load_data_array(dataset_csv)
    num_classes = len(labels)
    train_set = split_dataset(df, 'TRAIN')
    eval_set = split_dataset(df, 'VALIDATION')
    test_set = split_dataset(df, 'TEST')
    x_train, y_train = create_iterator(train_set,
                                       num_classes=num_classes,
                                       epochs=FLAGS.epochs,
                                       batch_size=FLAGS.batch_size,
                                       is_training=True)
    x_eval, y_eval = create_iterator(eval_set,
                                     num_classes=num_classes,
                                     epochs=FLAGS.epochs,
                                     batch_size=FLAGS.batch_size)
    x_test, y_test = create_iterator(test_set,
                                     num_classes=num_classes,
                                     epochs=FLAGS.epochs,
                                     batch_size=FLAGS.batch_size)

    logging.info('Build and compile model...')
    model = build_model(len(labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='output/board'),
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(filepath='output/best_checkpoint.hdf5',
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           period=1)
    ]

    logging.info('Training model...')
    model.fit(x=x_train,
              y=y_train,
              steps_per_epoch=(len(train_set) - 1) // FLAGS.batch_size,
              validation_data=(x_eval, y_eval),
              validation_steps=(len(eval_set) - 1) // FLAGS.batch_size,
              callbacks=callbacks,
              )

    logging.info('Evaluate model...')
    scores = model.evaluate(x=x_test, y=y_test, steps=(len(test_set) - 1) // FLAGS.batch_size)
    logging.info("test loss: {}, test accuracy: {}".format(scores[0], scores[1]))


if __name__ == "__main__":
    main()
