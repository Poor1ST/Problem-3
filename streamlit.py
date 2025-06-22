import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Load Model ---
# Use st.cache_resource for efficient loading
@st.cache_resource
def load_decoder_model():
    """Loads the trained Keras decoder model."""
    try:
        model = tf.keras.models.load_model('decoder.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

decoder_model = load_decoder_model()
LATENT_DIM = 20 # Must match the training script
NUM_CLASSES = 10

# --- 2. Streamlit Web App Interface ---
st.set_page_config(layout="wide")

st.title("Handwritten Digit Generation (TensorFlow)")
st.write("""
This app uses a **Conditional Variational Autoencoder (CVAE)** built with TensorFlow/Keras
to generate handwritten digits.
1.  **Select a digit** from the sidebar.
2.  Click the **Generate Images** button.
3.  The model will synthesize 5 new images of your chosen digit.
""")

# --- 3. User Input and Generation Logic ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Select a Digit (0-9):", list(range(10)))

if st.sidebar.button("Generate Images", type="primary"):
    if decoder_model is not None:
        st.subheader(f"Generated Images for Digit: {selected_digit}")

        with st.spinner("Generating..."):
            # Number of images to generate
            num_images = 5

            # 1. Sample random latent vectors
            random_latent_vectors = tf.random.normal(shape=[num_images, LATENT_DIM])

            # 2. Prepare one-hot encoded labels
            digit_labels = tf.keras.utils.to_categorical([selected_digit] * num_images, num_classes=NUM_CLASSES)

            # 3. Generate images using the decoder model
            generated_images = decoder_model.predict([random_latent_vectors, digit_labels])

            # Reshape for display
            generated_images = (generated_images * 255).astype(np.uint8)
            
            # Create a single image by concatenating horizontally
            cols = st.columns(num_images)
            for i in range(num_images):
                with cols[i]:
                    img = generated_images[i].reshape(28, 28)
                    st.image(img, caption=f"Sample {i+1}", width=100)
    else:
        st.error("Model could not be loaded. Cannot generate images.")
else:
    st.info("Select a digit and click the 'Generate Images' button to begin.")