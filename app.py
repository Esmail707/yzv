import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

### DO NOT EDIT IMPORTS ABOVE THIS LINE ###

# -----------------------------------------------------------------------------
# STEP 1: MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()


# -----------------------------------------------------------------------------
# STEP 2: UI LAYOUT
# -----------------------------------------------------------------------------
st.title("Image Captioning App")
st.markdown("**Name:** Ismail Al Hamdosh  |  **Student ID:** 150240903")
st.sidebar.header("Parameters Tuning")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.1, 0.1)
min_length = st.sidebar.slider("Min Length", 3, 20, 5, 1)
max_length = st.sidebar.slider("Max Length", 5, 30, 20, 1)
num_variations = st.sidebar.slider("Number of Variations", 1, 5, 1, 1)

# -----------------------------------------------------------------------------
# STEP 3: MAIN PIPELINE
# -----------------------------------------------------------------------------

st.markdown("Upload an image to generate captions using BLIP.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Optional starting text
    start_text = st.text_input("Optional starting text:")
    
    generate_button = st.button("Generate Caption")


if generate_button:
    with st.spinner("Generating caption(s)..."):
        # Preprocess image (and optional text)
        if start_text:
            inputs = processor(images=image, text=start_text, return_tensors="pt")
        else:
            inputs = processor(images=image, return_tensors="pt")

        # Loop for multiple variations
        for i in range(num_variations):
            out = model.generate(
                **inputs,
                do_sample=True,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=50
            )
            
            # Decode output
            caption_text = processor.decode(out[0], skip_special_tokens=True)
            
            # Display caption
            st.write(f"{i+1}. {caption_text}")
