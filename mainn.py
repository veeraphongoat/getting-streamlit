import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title
st.title('Microplastic Classification')

# Set Header
st.header('Please upload a picture')

# Load Model
from your_model import YourModelClass
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YourModelClass()  # Initialize an instance of your model
model.load_state_dict(torch.load('mobilenetv3_large_100_checkpoint_fold0.pt', map_location=device))

model.to(device)
model.eval()  # Ensure the model is in evaluation mode

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    class_name = ['Fighting', 'Hugging', 'Laughing', 'Dancing', 'Sitting', 'Sleeping', 'Running', 'Cycling', 'Calling', 'Drinking', 'Eating']

    if st.button('Prediction'):
        # Prediction class
        classname, probli = pred_class(model, image, class_name)

        st.markdown("## Prediction Result")

        # Get the index of the maximum value in probli
        max_index = np.argmax(probli)

        # Iterate over class_name and probli
        for i in range(len(class_name)):
            style = f"color: {'blue' if i == max_index else 'black'}"
            st.markdown(f"### <span style='{style}'>{class_name[i]}: {probli[i] * 100:.2f}%</span>", unsafe_allow_html=True)
