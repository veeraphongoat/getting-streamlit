import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np
from your_model import YourModelClass  # Import your model class

# Title and Description
st.title('Human Activity Recognition')
st.header('Please upload a picture')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define your model instance
model = YourModelClass()  # Replace with the actual model class

try:
    model.load_state_dict(torch.load('mobilenetv3_large_100_checkpoint_fold0.pt', map_location=device))
    model.eval()  # Set the model to evaluation mode
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Upload Image Section
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    class_name = ['Fighting', 'Hugging', 'Laughing', 'Dancing', 'Sitting', 'Sleeping', 'Running', 'Cycling', 'Calling', 'Drinking', 'Eating']

    # Prediction Section
    if st.button('Make Prediction'):
        # Prediction class
        probli = pred_class(model, image, class_name)
        
        st.subheader("Prediction Result")
        # Get the index of the maximum value in probli[0]
        max_index = np.argmax(probli[0])

        # Iterate over the class_name and probli lists
        for i in range(len(class_name)):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if i == max_index else None
            st.write(f"{class_name[i]} : {float(probli[0][i])*100:.2f}%", unsafe_allow_html=True, key=i)
