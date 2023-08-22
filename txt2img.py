import requests
import streamlit as st
from PIL import Image
import io
import PIL

API_URL = "https://api-inference.huggingface.co/models/LinoyTsaban/lora-xl-3d_icons-0.0001-5e-05-1500-1-None"
headers = {"Authorization": "Bearer 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"}

def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        image_bytes = response.content
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except PIL.UnidentifiedImageError as e:
            st.error("Error: Unable to identify the generated image.")
            st.text(f"UnidentifiedImageError: {e}")
            st.text("Image Bytes:")
            st.text(image_bytes)
            return None
    else:
        st.error("Error: Unable to generate image.")
        st.text(f"API Response Status Code: {response.status_code}")
        st.text("API Response Content:")
        st.text(response.content)
        return None

def main():
    st.set_page_config(page_title="Prompt to Image Generator", page_icon="🌄")
    st.title("Prompt to Image Generator")

    prompt = st.text_area("Enter a prompt:")

    if st.button("Generate Image"):
        image = generate_image(prompt)
        if image is not None:
            st.subheader("Generated Image:")
            st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
