from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
model = tf.keras.models.load_model(r"model/best_model2.h5", compile=False)

def import_and_predict(image, model):
    target_size = (256, 256)
    image = Image.open(image).convert("RGB")
    
    # Automatically center-crops and resizes
    image = ImageOps.fit(image, target_size, method=Image.LANCZOS, centering=(0.5, 0.5))
    
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            prediction = import_and_predict(file, model)
            class_names = ['AI', 'Real']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            prediction_text = f"{predicted_class} ({confidence:.2f})"
    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
