import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from htbuilder import (
    HtmlElement,
    div,
    br,
    hr,
    a,
    p,
    img,
    styles,
)
from htbuilder.units import percent, px


# Streamlit app title and description
st.set_page_config(
    page_title="Cancer Predicition",
    page_icon="https://cdn.the-scientist.com/assets/articleNo/70781/aImg/48671/cancer-cells-article-o.jpg",
)
st.title("Lung and Colon Cancer Prediction")
# Image uploader widget
st.sidebar.title("Welcome 👋")
selected = st.sidebar.selectbox(
    "Select page : ", ["Information page", "Model information", "Model predictions"]
)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(15376, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 5),
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize as per your model's requirements
        ]
    )
    try:
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return input_tensor

    except Exception as e:
        st.write("Upload relevant image")
        return None


def predict(test):
    model = torch.load("model.pt")
    model.eval()
    output = model(test)
    pred_class = np.argmax(output)
    labels = [
        "Colon Adenocarcinoma",
        "Colon Benign Tissue",
        "Lung Adenocarcinoma",
        "Lung Benign Tissue",
        "Lung Squamous Cell Carcinoma",
    ]
    # st.write("OUTPUT FROM MODEL : " , output)
    # st.write("Class: " ,pred_class)
    # st.write("Class name: " , labels[pred_class])
    return labels[pred_class]


if selected == "Information page":
    st.markdown(
        """<div style="text-align: justify">Histopathological analysis plays a crucial role in the  diagnosis and treatment of cancer. 
        In this project the ML-based system has been developed that can accurately classify 
        histopathological images of lung and colon cancer into five
        distinct classes: lung benign tissue, lung adenocarcinoma,
        lung squamous cell carcinoma, colon adenocarcinoma, and
        colon benign tissue. By automating this classification process, 
        healthcare professionals can save time, increase efficiency, and improve the accuracy of cancer diagnosis.<br><br>
<h3>Dataset information</h3>
The dataset used for this project is the Lung and Colon 
Cancer Histopathological Images dataset, also known as LC25000. This dataset contains 25,000 histopathological images 
of size 768 x 768 pixels in JPEG format. The dataset was obtained from HIPAA compliant and validated sources, consisting of 750 original 
images of lung tissue and 500 original images of colon tissue. The images were augmented to reach a total of 25,000 images.
The dataset contains images with 5 classes, including lung and colon cancer and healthy samples. 
Each class contains 5,000 images of the following histologic entities: colon adenocarcinoma, 
benign colonic tissue, lung adenocarcinoma, lung squamous cell carcinoma, and normal lung tissue. The 
dataset is publicly available on Kaggle and was assembled by Andrew A. Borkowski and his associates.<br>
<img src = "https://github.com/Snehee2901/CancerApp/blob/main/lungaca1.jpeg">
</div>""",
        unsafe_allow_html=True,
    )
# Display the uploaded image
if selected == "Model predictions":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("")

        with col2:
            st.image(
                image.resize((200, 200)),
                caption="Uploaded Image",
                use_column_width=False,
            )

        with col3:
            st.write("")

        tensor = preprocess(image)
        if tensor != None:
            with torch.no_grad():
                predicted_class = predict(tensor)

            # Display the prediction
            st.write(f"The image you uploaded seems to be of : {predicted_class}")
        # Optionally, you can display the uploaded image as well
    else:
        st.write("No image uploaded yet.")


def imag(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def links(l, text, **style):
    return a(_href=l, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        width=percent(100),
        color="black",
        text_align="center",
        height=0,
        opacity=1,
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2),
    )
    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.sidebar.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)
    st.sidebar.markdown(foot, unsafe_allow_html=True)


def footer():
    myargs = [
        "<b>Technologies</b>:",
        br(),
        " Python 3.10 ",
        links(
            "https://www.python.org/",
            imag(
                "https://i.imgur.com/ml09ccU.png",
                width=px(18),
                height=px(18),
                margin="0em",
            ),
        ),
        ", Streamlit ",
        links(
            "https://streamlit.io/",
            imag(
                "https://image.pngaaa.com/798/5084798-middle.png",
                width=px(24),
                height=px(25),
                margin="0em",
            ),
        ),
        br(),
        "<b>Made by Snehee Patel</b>",
        br(),
        "Connect with me on :",
        br(),
        links(
            "https://github.com/Snehee2901",
            imag(
                "data:image/png;base64,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",
                width=px(24),
                height=px(25),
                margin="0.25em",
            ),
        ),
        links(
            "https://www.linkedin.com/in/SneheePatel/",
            imag(
                "https://image.similarpng.com/very-thumbnail/2020/07/Linkedin-logo-on-transparent-Background-PNG-.png",
                width=px(24),
                height=px(25),
                margin="0.25em",
            ),
        ),
        links(
            "https://www.kaggle.com/sneheepatel",
            imag(
                "https://thumbnail.imgbin.com/5/4/0/imgbin-kaggle-predictive-modelling-data-science-business-predictive-analytics-4Mh2z1pTSSFjmKfX09tHiQrz7_t.jpg",
                width=px(34),
                height=px(25),
                margin="0.25em",
            ),
        ),
        br(),
    ]
    layout(*myargs)


footer()
