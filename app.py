import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


# Modeli yükleyin
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1) #first conv. layer
        self.relu=nn.ReLU() #relu 
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2) #maxpooling
        self.flatten=nn.Flatten() #flatten 
        self.fc1 = nn.Linear(32*14*14,10)

    def forward(self,x): 
        x = self.conv1(x) 
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

model = CNN()
model.load_state_dict(torch.load('/Users/erenozkan/Desktop/github_1/handwritten-number-prediction/model_weights.pth'))
st.set_page_config(
    page_title="Handwritten Number Prediction",
    layout="wide",
    initial_sidebar_state="expanded",

)

#st.title("Handwritten Number Prediction")
st.markdown("<h1 style='text-align: center;'>Handwritten Number Prediction</h1>", unsafe_allow_html=True)
st.markdown("* <h5>This application was created with the CNN model. This model was trained on 60000 images from the MNIST dataset.</h5>", unsafe_allow_html=True)
st.markdown("* <h5> Draw a single number with your mouse and click Predict button.</h5>", unsafe_allow_html=True)
st.markdown("* For more information, check the [training process](https://github.com/ozkaner/MNIST) of the CNN model.")




#canvas area
canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#ffffff",
    background_color="#000000",
    height=240,
    width=240,
    drawing_mode="freedraw",
    key="canvas",
)

# Tahmin butonu
if st.button("Predict"):
    if canvas_result.image_data is not None:
        drawing = canvas_result.image_data

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        reshaped_tensor = transform(drawing)
        input = reshaped_tensor.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

        st.write(f"Number is {predicted_class}")

        



