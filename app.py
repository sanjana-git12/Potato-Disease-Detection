import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("potato_model.keras")

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

IMAGE_SIZE = (256, 256)

def predict(image):

    image = image.resize(IMAGE_SIZE)
    img = np.array(image)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence



st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔"
)

st.title("Potato Disease Detection")
st.write(
    "Upload a potato leaf image to predict whether it is Healthy, Early Blight, or Late Blight."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            disease, confidence = predict(image)

        st.success(f"Prediction: **{disease}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        if disease == "Healthy":
            st.success("The potato leaf appears healthy.")

        elif disease == "Early Blight":
            st.warning(
                "⚠ Early Blight detected.\n\n"
                "Recommendation:\n"
                "- Remove infected leaves\n"
                "- Use recommended fungicides\n"
                "- Avoid overhead irrigation"
            )

        else:
            st.error(
                "Late Blight detected.\n\n"
                "Recommendation:\n"
                "- Remove infected plants\n"
                "- Apply fungicides immediately\n"
                "- Prevent water accumulation"
            )