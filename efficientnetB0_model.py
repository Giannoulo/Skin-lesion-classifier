import os
import tensorflow as tf


def load_EfficientnetB0(rootPath):
    from keras.models import load_model
    import efficientnet.keras

    model_path = os.path.join(rootPath, 'static/models')
    model = load_model(os.path.join(model_path, '2243220.h5'))

    return model


def predict_class(filepath, model, graph):
    from keras.preprocessing import image

    class_dict = {1: 'Actinic keratosis', 2: 'Basal cell carcinoma', 3: 'Benign keratosis', 4: 'Dermatofibroma',
                  5: 'Melanoma', 6: 'Melanocytic nevus', 7: 'Squamous cell carcinoma', 8: 'Vascular lesion'}
    class_predictions = {}

    # Read the image and resize it
    img = image.load_img(filepath, target_size=(224, 224))
    # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
    # Reshape
    x = x.reshape((1,) + x.shape)
    x = x/255.
    with graph.as_default():
        prediction = model.predict([x])
        for i in range(len(prediction[0])):
            class_predictions[class_dict[i+1]] = round(prediction[0][i]*100, 2)
    return class_predictions
