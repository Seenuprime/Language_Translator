import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Welcome to the language translator")

GROQ_API_KEY = st.secrets['GROQ_API_KEY']

llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=GROQ_API_KEY)

def prompt(content, language):
    template = """
            You are a language translator.

            1. Identify and state the language of the following text: {content}.  
            2. Translate the same text into {language}.

            Output format:  
            Language detected: [Detected Language]

            Translated to {language}:
            [Translation]
        """


    prompt_template = PromptTemplate(
        input_variables = ['content', 'language'],
        template=template
    )
    return prompt_template.format(content=content, language=language)

content = st.text_area("Enter the content to translate: ")
language = st.text_input("Enter the language to translate: ")


if st.button("Submit"):
    result = llm.invoke(prompt(content, language))
    st.write(result.content)
