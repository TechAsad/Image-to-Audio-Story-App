from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import GooglePalm

import requests
import os
import streamlit as st

os.environ["GOOGLE_API_KEY"] = "AIzaSyD29fEos3V6S2L-AGSQgNu03GqZEIgJads"

llm = GooglePalm(temperature=0.7)

# Image to text using Hugging Face API
def image2text(filename):
    API_URL1 = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers1 = {"Authorization": "Bearer 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"}
    
    with open(filename, "rb") as f:
        data = f.read()
        
    response = requests.post(API_URL1, headers=headers1, data=data)
    response_data = response.json()
    
    if "generated_text" in response_data[0]:
        generated_text = response_data[0]["generated_text"]
        return generated_text
    else:
        return "Image caption not available."

def generate_story(scenario):
    template = """
    You are a story teller;
    you can generate a creative, meaningful, scenario-based, funny story based on a sample narrative, the story should not be more than 150 words;

    CONTEXT: {scenario}
    STORY: 
    """
    prompt = PromptTemplate(template=template, input_variables=['scenario'])
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    return story

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        scenario = image2text(uploaded_file.name)
        st.subheader("Image Details:")
        st.write(scenario)

        story = generate_story(scenario)
        st.subheader("Story:")
        st.write(story)

        text2speech(story)
        st.subheader("Generated Audio:")
        st.audio("audio.flac", format="audio/flac")

        # Add a download link for the audio
        st.subheader("Download Audio:")
        with open("audio.flac", "rb") as audio_file:
            st.download_button(label="Download Audio", data=audio_file, file_name="generated_audio.flac", mime="audio/flac")

if __name__ == "__main__":
    main()
