import requests
import streamlit as st
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/LinoyTsaban/lora-xl-3d_icons-0.0001-5e-05-1500-1-None"
headers = {"Authorization": "Bearer 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"}

def generate_image(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    return image

def main():
    st.set_page_config(page_title="Prompt to Image Generator", page_icon="🌄")
    st.title("Prompt to Image Generator")

    prompt = st.text_area("Enter a prompt:")

    if st.button("Generate Image"):
        image = generate_image(prompt)
        st.subheader("Generated Image:")
        st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
