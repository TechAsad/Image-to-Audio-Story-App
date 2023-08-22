from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import GooglePalm

import requests
import os
import streamlit as st


os.environ["GOOGLE_API_KEY"] = "AIzaSyD29fEos3V6S2L-AGSQgNu03GqZEIgJads"
os.environ ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = GooglePalm(temperature = 0.7)


#image to text

def image2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-large")

    text = image_to_text(
        url)[0]['generated_text']
    
    print(text)
    return(text)

#story teller
def generate_story(scenario):
    template = """"
    You are a story teller;
    you can generate a creative fun story based on a sample narrative, the story should not be more than 100 words;

    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template = template, 
        input_variables = ['scenario']
    )
    story_llm = LLMChain(llm=llm, prompt = prompt, verbose = True)

    story = story_llm.predict(scenario = scenario)

    print(story)
    return(story)

#text to speech

def text2speech(message):
     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
     headers = {"Authorization": "Bearer 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"}
     payloads = {
          "inputs":message
     }
     response = requests.post(API_URL, headers = headers, json= payloads)
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
        st.image(uploaded_file, caption="Uploaded Image",
                 use_column_width= True)
        
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
