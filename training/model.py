from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Dense


def build_model(num_classes: int,
                input_shape: () = (224, 224),
                pooling: str = 'avg'):
    base_model = Xception(include_top=False, input_shape=(*input_shape, 3), pooling=pooling, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output

    logits = Dense(num_classes, name='scores', activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=logits)
    return model
