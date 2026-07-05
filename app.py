import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_model.keras")

model = load_model()


CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

st.title("Potato Disease Detection")
st.write("Upload a potato leaf image to detect the disease.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((256, 256))

    img_array = np.array(image)


    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])

    confidence = np.max(prediction[0]) * 100

    st.subheader("Prediction")
    st.success(CLASS_NAMES[predicted_index])

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")

    
    with st.expander("Model Output"):
        st.write(prediction)

    if CLASS_NAMES[predicted_index] == "Healthy":
        st.success("The potato leaf appears healthy.")

    elif CLASS_NAMES[predicted_index] == "Early Blight":
        st.warning(
            "Early Blight detected.\n\n"
            "• Remove infected leaves.\n"
            "• Apply recommended fungicide.\n"
            "• Avoid overhead irrigation."
        )

    else:
        st.error(
            "Late Blight detected.\n\n"
            "• Remove infected plants immediately.\n"
            "• Apply fungicide.\n"
            "• Improve field drainage."
        )
