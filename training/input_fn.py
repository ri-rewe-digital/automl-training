import tensorflow as tf

input_size = 224


def one_hot_encode(image, label, num_classes: int):
    return image, tf.one_hot(label, num_classes)


def load_img(filename: str, label, size: int):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image, [size, size])
    return resized_image, label


def input_fn(filenames: [],
             labels: [],
             num_classes: int,
             batch_size: int,
             epochs: int,
             is_training: bool, ):

    num_entries = len(filenames)
    assert num_entries == len(labels), 'Length of labels is not equal to image list'

    load_fn = lambda f, l: load_img(f, l, input_size)
    one_hot_fn = lambda f, l: one_hot_encode(f, l, num_classes)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_entries)
                   .repeat(epochs)
                   .map(load_fn, num_parallel_calls=4)
                   .map(one_hot_fn, num_parallel_calls=4)
                   .batch(batch_size)
                   .prefetch(2)
                   )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(load_fn, num_parallel_calls=4)
                   .map(one_hot_fn, num_parallel_calls=4)
                   .batch(batch_size)
                   .prefetch(1)
                   )

    # Create re-initializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    tf.keras.backend.get_session().run(iterator_init_op)  # initiate the iterator in the session
    images, labels = iterator.get_next()
    return images, labels, iterator_init_op
