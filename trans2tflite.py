import tensorflow as tf
import numpy as np
import time

for i in range(1,7):
    model = tf.keras.models.load_model(f"./Models/Exit_Model_{i}.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f"./TFLITE_Models/Exit_Model_{i}.tflite","wb").write(tflite_model)