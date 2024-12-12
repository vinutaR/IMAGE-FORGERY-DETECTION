import numpy as np
from tensorflow.keras.models import load_model
from ela import convert_to_ela_image


def prepare_image(fname):
    image_size = (128, 128)
    return (
        np.array(convert_to_ela_image(fname[0], 90).resize(image_size)).flatten()
        / 255.0
    )  # return ela_image as a numpy array


def predict_result(fname):
    model = load_model("image-forgery-detection-main\trained_model.h5")  # load the trained model
    class_names = ["Forged", "Authentic"]  # classification outputs
    test_image = prepare_image(fname)
    test_image = test_image.reshape(-1, 128, 128, 3)

    y_pred = model.predict("C:\Users\H P\Desktop\image-forgery-detection-main\image-forgery-detection-main\saved_image.jpg")
    y_pred_class = round(y_pred[0][0])

    prediction = class_names[y_pred_class]
    if y_pred <= 0.5:
        confidence = f"{(1-(y_pred[0][0])) * 100:0.2f}"
    else:
        confidence = f"{(y_pred[0][0]) * 100:0.2f}"
    return (prediction, confidence)
