import keras
from tensorflow import keras 
import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
import base64
from dotenv import load_dotenv 
import os


st.set_page_config(
    page_title="Домашняя страница",
    page_icon="👋",
    layout='wide'
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('D:\\Waste-Classifier-Sustainability-App\\sdg goals\\background2.jpg')

def add_logo(png_file):
    bin_str = get_base64(png_file)
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("data:image/png;base64,%s");
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: " Нгуен Ван Мань - ИКБО-04-20";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 20px;
                position: relative;
                top: 100px;
            }
        </style>
        """ % bin_str,
        unsafe_allow_html=True,
    )
add_logo('D:\machineLearning\Machine_Learning\WebSite\images\logo_mirea.png')


def classify_waste(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model("keras_model.h5", compile=False)
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = img.convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    #Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    return class_name, confidence_score





st.title("Система классификации отходов")

choice = st.sidebar.selectbox("Выберите свой выбор", ["Камера", "Картина"])
# input_img = st.file_uploader("Введите свое изображение", type=['jpg', 'png', 'jpeg'])
col1, col2 = st.columns(2)


    
with col1:
    if choice == "Камера":
        input_img = st.camera_input("Сделать фото")
    elif choice == "Картина":
        input_img = st.file_uploader("Введите свое изображение", type=['jpg', 'png', 'jpeg'])

# with col2:
    # input_img = st.file_uploader("Введите свое изображение", type=['jpg', 'png', 'jpeg'])



if input_img is not None:
    if st.button("Классифицировать"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.info("Ваше загруженное изображение")
            st.image(input_img, use_column_width=True)
        with col2:
            st.info("Результаты классификации")
            image_file = Image.open(input_img)
            print(type(image_file))
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1,1])
            if label == "0 cardboard\n":          
                with col4:
                    st.success("Изображение классифицируется как КАРТОН.")                  
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/13.png", use_column_width=True)
                with col5:
                    st.success(confidence_score)
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "1 plastic\n":
                with col4:
                    st.success("Изображение классифицируется как ПЛАСТИК.") 
                    st.image("sdg goals/6.png", use_column_width=True)
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.success(confidence_score)
                    st.image("sdg goals/14.png", use_column_width=True)
                    st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "2 glass\n":
                with col4:
                    st.success("Изображение классифицируется как СТЕКЛО.") 
                    st.image("sdg goals/12.png", use_column_width=True)
                with col5:
                    st.success(confidence_score)
                    st.image("sdg goals/14.png", use_column_width=True)
            elif label == "3 metal\n":
                with col4:
                    st.success("Изображение классифицируется как МЕТАЛЛ.") 
                    st.image("sdg goals/3.png", use_column_width=True)
                    st.image("sdg goals/6.png", use_column_width=True)
                with col5:
                    st.success(confidence_score)
                    st.image("sdg goals/12.png", use_column_width=True)
                    st.image("sdg goals/14.png", use_column_width=True) 
            else:
                st.error("Изображение не относится к какому-либо соответствующему классу.")
        #with col3:
            #st.info("GPT text")
            # result = generate_carbon_footprint_info(label)
            # st.success(result)





# def generate_carbon_footprint_info(label):
#     label = label.split(' ')[1]
#     print("label: ", label)
#     response = client.completions.create(
#     model="text-davinci-003",
#     prompt="What is the approximate Carbon emission or carbon footprint generated from "+label+"? I just need an approximate number to create awareness. Elaborate in 100 words.",
#     temperature=0.7,
#     max_tokens=600,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']


