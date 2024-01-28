import streamlit as st
from torch import optim
from PIL import Image
from torchvision import models
from nst import device, NSTCost, img_size

model = models.vgg19(pretrained=True).features.requires_grad_(False).eval().to(device)
layers = list(model.children())

content_layer_idx = 35
style_layers_idx = [1, 6, 11, 20, 29]
style_layer_weights = {
    1: 0.2,
    6: 0.2,
    11: 0.2,
    20: 0.2,
    29: 0.2,
}

alpha = 1
beta = 2

col1, col2 = st.columns(spec=2, gap='medium')
col1.write('<h3>Upload a style image</h3>', unsafe_allow_html=True)
col1.write(f'{device}')
style_image = col1.file_uploader('Style', type=['png', 'jpg'], label_visibility='hidden')
col2.write('<h3>Upload a content image</h3>', unsafe_allow_html=True)
content_image = col2.file_uploader('Content', type=['png', 'jpg'], label_visibility='hidden')
sub = col2.button('Transfer')

nst_cost = None  # Initialize outside the conditional block

if content_image and style_image:
    content_image = Image.open(content_image).resize(img_size)
    style_image = Image.open(style_image).resize(img_size)
    nst_cost = NSTCost(
        content_image,
        style_image,
        model,
        layers,
        content_layer_idx,
        style_layers_idx,
        style_layer_weights,
        alpha,
        beta,
        optim.Adam,
        {"lr": 0.01},
        device,
    )

if sub and nst_cost:
    img = nst_cost.fit(1000)
    col1.image(img)

