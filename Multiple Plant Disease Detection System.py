import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import random
from io import BytesIO

# Load the model once at the beginning
model = tf.keras.models.load_model(r"C:\Users\AMAN KUMAR\Downloads\MPDDIV4\MobileNetV2 Model\mobnet_fine_tuned_model.keras")

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    try:
        # Load and preprocess the uploaded image
        image = Image.open(test_image)
        image = image.convert('RGB')  # Ensure image is in RGB mode
        image = image.resize((224, 224))  # Ensure the target size matches your model input
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
        input_arr = input_arr / 255.0  # Normalize the image (assuming the model was trained with normalization)

        # Predict the class of the image
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return the index of the class with the highest probability
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Convert Image to Base64
def image_to_base64(image_path):
    """Converts image to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Convert Image to Base64
def image_to_base64(image_s):
    """Converts image to base64 string"""
    with open(image_s, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Sidebar
st.sidebar.title("MENU")
app_mode = st.sidebar.selectbox("", ["Home", "About"])

# Main Page: Home
if app_mode == "Home":
    st.header("PLANT DISEASE DETECTION SYSTEM")

    # Image Paths
    image_paths = [
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/apple.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/corn.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/grape.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/potato.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/tomato.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/apple2.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/corn2.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/grape1.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/potato1.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/tomato1.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/apple.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/corn.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/grape3.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/potato2.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/static/tomato.jpg",
    ]

    image_s = [
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/Internet_Test_Image/apple_hy.jpeg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/Internet_Test_Image/apple_sb.jpeg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/Internet_Test_Image/Corn-cr.jpg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/Internet_Test_Image/grape_br.jpeg",
        "C:/Users/AMAN KUMAR/Downloads/MPDDIV4/MobileNetV2 Model/Internet_Test_Image/apple_cedar_ar.jpeg",
    ]
    
    # First image (Apple) always shown first
    first_image_path = image_paths[0]
    img_base64 = image_to_base64(first_image_path)
    first_image_html = f'<img src="data:image/png;base64,{img_base64}" alt="Image" style="height: 150px; border-radius: 10px; margin-right: 10px;">'

    # Randomizing remaining images
    remaining_images = image_paths[1:]
    random.shuffle(remaining_images)
    
    # Creating HTML string for scrolling images
    images_html = first_image_html  # Start with the first image
    for image_path in remaining_images:
        img_base64 = image_to_base64(image_path)
        images_html += f'<img src="data:image/png;base64,{img_base64}" alt="Image" class="image" style="height: 150px; border-radius: 10px; margin-right: 10px;">'

    scrolling_html = f"""
    <div style="position: relative; height: 200px; overflow: hidden;">
        <style>
            .scrolling {{
                position: relative;
                height: 100%;
                display: flex;
                animation: scroll 15s linear infinite;
            }}

            @keyframes scroll {{
                0% {{
                    transform: translateX(-100%);  /* Start from the right */
                }}
                100% {{
                    transform: translateX();  /* End at the left */
                }}
            }}
        </style>
        <div class="scrolling">
            {images_html}
        </div>
    </div>
    """
    
    # Display the scrolling images
    st.components.v1.html(scrolling_html, height=180)

    # Image Upload or Select
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    # Provide a "None" option along with test images
    st.write("Choose an image from the test options below or upload your own:")
    test_image_option = st.selectbox(
        "Select Test Image:", 
        ["None"] + image_s[:5],  # Add "None" as the first option
        format_func=lambda x: "None" if x == "None" else x.split("/")[-1]
    )

    # Only set the test_image if the user selects something other than "None"
    if test_image_option != "None":
        test_image = test_image_option

    if test_image is not None and test_image != "None":
        # Display the image in a smaller box (limit the size to a specific dimension)
        st.image(test_image, width=200, use_container_width=False)  # Set width to 200px for smaller display

    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        # Replace with your actual class labels
        class_name = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___healthy'
        ]

        if result_index is not None:
            st.success(f"Model Prediction: It's a {class_name[result_index]}")
        else:
            st.error("Prediction failed.")

# About Project
elif app_mode == "About":
    st.header("About")
    
    # Create a dropdown or radio button to switch between sub-sections in the About page
    about_section = st.radio(
        "Choose a section to explore:",
        ["About the Project", "About the Developers"]
    )
    
    if about_section == "About the Project":
        st.subheader("About the Project")
        st.markdown("""#### Project Overview
The Multiple Plant Disease Detection System leverages cutting-edge deep learning techniques to accurately identify plant diseases through image analysis. Using a fine-tuned MobileNetV2 model, this system is designed to help farmers, agricultural experts, and plant enthusiasts quickly diagnose plant diseases and take timely action. The goal is to reduce crop losses, promote sustainable farming practices, and assist in early disease detection.

#### Key Features
* __Accurate Disease Detection:__  Identifies diseases in plant leaves from a variety of crops including apples, corn, tomatoes, and more.\n
* __User-Friendly Interface:__  Easily upload images of plant leaves to get real-time disease predictions.\n
* __Machine Learning Model:__  Built using a fine-tuned MobileNet V2 model, trained on a large dataset of plant leaf images.\n
* __Multiple Crop Support:__  Capable of detecting diseases in various crops with 19 distinct classes.\n
                      
#### Project Goal
The primary objective of this project is to:

__1. Support Farmers:__ Provide a cost-effective solution for early disease detection.\n
__2. Improve Crop Yield:__ Reduce the time between disease occurrence and identification.\n
__3. Promote Sustainability:__ Encourage efficient resource use by targeting specific diseases.\n
                    
#### Technology Stack
* __Deep Learning Framework:__ TensorFlow and Keras for model training and inference.\n
* __Model Architecture:__ MobileNet V2, a powerful convolutional neural network for image classification.\n
* __Frontend Interface:__ Designed with a user-friendly layout for easy interaction.\n
* __Programming Language:__ Python, used for both model development and deployment.\n
* __Optimization:__ Early stopping and model checkpointing to ensure the best performance during training.\n

#### About the Dataset
The dataset used in this project is a comprehensive collection of plant leaf images categorized into 19 classes, covering both healthy and diseased leaves.\n

Training Images: 35669\n
Validation Images: 8917\n
Test Images: 25 (for performance evaluation) The dataset includes a wide variety of plant diseases and conditions, making it robust for real-world applications.
                            
#### Achieved Accuracy
__I. Training Accuracy: 99.67%__\n
__II. Validation Accuracy: 99.24%__\n
The model achieves excellent results in detecting plant diseases, showcasing the power of deep learning in agriculture.
                    
#### Future Enhancements
__Multi-language Support:__ To cater to users globally, we plan to add multilingual capabilities.\n
__Broader Crop Coverage:__ Adding support for more crops and diseases.\n
__Mobile Application:__ Developing a mobile app for on-the-go disease detection.\n
#### Conclusion
We hope this project inspires innovative solutions in the field of agriculture and plant health. Whether you're a farmer, researcher, or tech enthusiast, this tool is designed to empower you with the power of AI for a better tomorrow.

Together, we can make strides in sustainable farming and secure healthier crops for the future. Thank you for exploring this system, and we look forward to your feedback and ideas to improve and expand its capabilities! 🌱
        """)
        
    elif about_section == "About the Developers":
        st.subheader("About the Developers")
        st.markdown("""
            This project was developed by a team of dedicated students with a shared interest in leveraging technology to tackle real-world challenges in agriculture. Our goal was to build a plant disease detection system using deep learning to help farmers detect plant diseases quickly and accurately.""")


        st.markdown("__Team Members:__")

        st.markdown("__Aman Kumar - Project Lead & Developer__")
        st.markdown("""Aman took the lead in the development of the project, overseeing the technical execution. He focused on training and fine-tuning the Inception V3 model for plant disease detection and integrating the model with the Streamlit interface. Aman ensured the overall flow of the application worked seamlessly from image upload to disease prediction.""")

        st.markdown("__Harsh Pratap Singh - Frontend Developer__")
        st.markdown("""Harsh Pratap Singh was responsible for creating the user interface and ensuring an intuitive and smooth experience for users. They designed the Streamlit frontend, providing a simple yet effective way for users to upload images and receive disease predictions.""")

        st.markdown("__MD. Badrul Hoda - Tester & Quality Assurance__")
        st.markdown("""MD. Badrul Hoda was responsible for testing the system, ensuring that the application performed as expected. They conducted thorough testing to identify bugs, glitches, and user experience issues, providing feedback to the team to enhance the application's quality, stability, and usability.""")

        st.markdown("__Shubham Gupta - Support Specialist__")
        st.markdown("""Shubham Gupta played a crucial supporting role throughout the development of the project. Their responsibility was to assist in various tasks, including dataset organization, documentation, and user interface enhancements. They also helped with troubleshooting and ensuring the smooth progress of the project by assisting all team members as needed.
                """)
        st.subheader("Acknowledgements")
        st.markdown("""We would like to express our gratitude to the researchers and contributors who provided the plant disease dataset. Their work was essential to the success of this project. Additionally, we thank our mentors for their continuous support and guidance throughout the development process.
                """)
        st.markdown("""
        <div style="text-align: right;">
        <P>- Team Members</p>
        </div>
        """, unsafe_allow_html=True)
