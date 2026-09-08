"""
Streamlit app: Satellite Land Cover Classifier
Upload a Sentinel-2 style satellite image patch and get a predicted
land cover class (EuroSAT categories) with confidence scores.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils import CLASS_NAMES, CLASS_DESCRIPTIONS, preprocess_image

st.set_page_config(
    page_title="Land Cover Classifier",
    page_icon="🛰️",
    layout="centered",
)

MODEL_PATH = os.path.join("model", "landcover_cnn.h5")


@st.cache_resource
def load_model():
    """Load the trained Keras model. Returns None if it hasn't been trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, pil_image: Image.Image):
    x = preprocess_image(pil_image)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return [(CLASS_NAMES[i], float(probs[i])) for i in order]


def main():
    st.title("🛰️ Satellite Land Cover Classifier")
    st.caption(
        "A CNN trained on the EuroSAT (Sentinel-2) dataset to classify "
        "10 land cover types from a satellite image patch — built after "
        "completing ISRO IIRS's *AI/ML for Geodata Analytics* course."
    )

    model = load_model()

    with st.sidebar:
        st.header("About")
        st.write(
            "Model: small CNN (3 conv blocks + GAP)\n\n"
            "Dataset: EuroSAT RGB, 10 classes, 64x64 Sentinel-2 patches\n\n"
            "Framework: TensorFlow / Keras"
        )
        st.header("Classes")
        for c in CLASS_NAMES:
            st.markdown(f"- **{c}** — {CLASS_DESCRIPTIONS[c]}")

    if model is None:
        st.warning(
            "No trained model found yet at `model/landcover_cnn.h5`.\n\n"
            "Run `train_model_colab.ipynb` in Google Colab (free GPU), "
            "then download the saved model file into this project's "
            "`model/` folder and redeploy. The app UI below still works "
            "for previewing uploaded images in the meantime."
        )

    uploaded = st.file_uploader(
        "Upload a satellite image patch (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    sample_col1, sample_col2 = st.columns(2)
    use_sample = sample_col1.button("Try a sample image")

    image = None
    if uploaded is not None:
        image = Image.open(uploaded)
    elif use_sample:
        sample_dir = "sample_images"
        samples = [f for f in os.listdir(sample_dir)] if os.path.isdir(sample_dir) else []
        if samples:
            image = Image.open(os.path.join(sample_dir, samples[0]))
        else:
            st.info("Add a few example patches to sample_images/ to enable this button.")

    if image is not None:
        st.image(image, caption="Input image", use_container_width=True)

        if model is not None:
            with st.spinner("Classifying..."):
                results = predict(model, image)

            top_class, top_conf = results[0]
            st.success(f"Predicted: **{top_class}** ({top_conf*100:.1f}% confidence)")
            st.caption(CLASS_DESCRIPTIONS[top_class])

            df = pd.DataFrame(results, columns=["Class", "Probability"]).set_index("Class")
            st.bar_chart(df)
        else:
            st.info("Upload a model to model/landcover_cnn.h5 to get real predictions.")

    st.divider()
    st.caption(
        "Built as a practical follow-on to the ISRO IIRS AI/ML for Geodata "
        "Analytics course. Source code on GitHub — see README for training "
        "and deployment instructions."
    )


if __name__ == "__main__":
    main()
