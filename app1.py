import streamlit as st 
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('API_KEY')

def generate_carbon_footprint_info(label):
    label = label.split(' ')[1]
    print("label: ", label)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="What is the approximate Carbon emission or carbon footprint generated from "+label+"? I just need an approximate number to create awareness. Elaborate in 100 words.",
    temperature=0.7,
    max_tokens=600,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

st.info("GPT text")
label = "1 plastic"
result = generate_carbon_footprint_info(label)
st.success(result)






# import streamlit as st 
# import os
# import openai


# openai.api_key = api_key

# start = "Your are a AI Search Engine, answer the following query with a witty answer and include validated facts only."
# label = "1 plastic"

# def generate_carbon_footprint_info(label):
#     label = label.split(' ')[1]
#     print(label)
#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt="What is the approximate Carbon emission or carbon footprint generated from "+label+"? I just need an approximate number to create awareness. Elaborate in 100 words.\n",
#     temperature=0.7,
#     max_tokens=600,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']



# result = generate_carbon_footprint_info(label)

# st.success(result)