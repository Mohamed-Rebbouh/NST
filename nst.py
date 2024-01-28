from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from rich.progress import Progress
from torch import nn, optim
from torchvision import models, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 512, 512

def compute_content_cost(content_output, generated_output):
    # shape = (n_c, n_h, n_w)
    if content_output.shape != generated_output.shape:
        raise Exception("content_output and generated_output have different shapes")
    n_c, n_h, n_w = content_output.shape
    cft = 1 / (4 * n_h * n_w * n_c)
    return cft * ((content_output - generated_output) ** 2).sum()


def compute_layer_style_cost(style_output, generated_output):
    # shape = (n_c, n_h, n_w)
    if style_output.shape != generated_output.shape:
        raise Exception("style_output and generated_output have different shapes")
    n_c, n_h, n_w = style_output.shape
    style_unrolled = style_output.view(n_c, -1)
    generated_unrolled = generated_output.view(n_c, -1)
    # compute gram matrix
    style_gram = style_unrolled.matmul(style_unrolled.transpose(0, 1))
    generated_gram = generated_unrolled.matmul(generated_unrolled.transpose(0, 1))
    cft = 1 / (2 * n_c * n_h * n_w) ** 2
    return cft * ((style_gram - generated_gram) ** 2).sum()


class NSTCost:
    def __init__(
        self,
        content_img,
        style_img,
        model,
        layers,
        content_layer_idx,
        style_layers_idx,
        style_layer_weights,
        alpha,
        beta,
        optimizer,
        optimizer_kwargs,
        device,
    ):
        self.device = device
        self.content_img = content_img
        self.style_img = style_img
        self.model = model
        self.layers = layers
        self.content_layer_idx = content_layer_idx
        self.content_layer_activation = None
        self.style_layers_idx = style_layers_idx
        self.style_layers_activation = {i: None for i in style_layers_idx}
        self.style_layer_weights = style_layer_weights
        self.alpha = alpha
        self.beta = beta
        self.register_forward_hooks()
        self.normalizer = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        self.denormalizer = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        self.to_tensor = transforms.ToTensor()
        self.style_resizer = transforms.Resize(img_size)
        self.style_outputs = self.get_style_outputs(style_img)
        self.content_output = self.get_content_output(content_img)
        self.generated_tensor = self.get_generated_tensor()
        self.optimizer = optimizer((self.generated_tensor,), **optimizer_kwargs)

    def get_generated_tensor(self):
        generated_tensor = (
            self.get_content_tensor(self.content_img).clone().to(self.device)
        )
        # add noise
        generated_tensor.add_(torch.rand_like(generated_tensor))
        generated_tensor = generated_tensor.requires_grad_(True)
        return generated_tensor

    def get_style_tensor(self, style_img):
        style_tensor = self.normalizer(
            self.style_resizer(self.to_tensor(style_img))
        ).to(self.device)
        style_tensor.unsqueeze_(0)
        return style_tensor

    def get_style_outputs(self, style_img):
        style_tensor = self.get_style_tensor(style_img)
        self.model(style_tensor)
        style_outputs = self.style_layers_activation.copy()
        return style_outputs

    def get_content_tensor(self, content_img):
        content_tensor = self.normalizer(self.to_tensor(content_img)).to(device)
        content_tensor.unsqueeze_(0)
        return content_tensor

    def get_content_output(self, content_img):
        content_tensor = self.get_content_tensor(content_img)
        self.model(content_tensor)
        content_output = self.content_layer_activation
        return content_output

    def register_style_image(self, style_image):
        self.model(style_image)

    def content_forward_hook(self, module, input, output):
        # print('content hook called')
        self.content_layer_activation = output

    def style_forward_hook(self, module, input, output):
        # print('style hook called')
        self.style_layers_activation[module.idx] = output

    def register_forward_hooks(self):
        idx = self.content_layer_idx
        self.layers[idx].register_forward_hook(self.content_forward_hook)
        for idx in self.style_layers_idx:
            self.layers[idx].idx = idx  # monkey patch the idx attribute
            self.layers[idx].register_forward_hook(self.style_forward_hook)

    def gather_generated_outputs(self, generated_image):
        outputs = {}
        self.model(generated_image)
        outputs["content"] = self.content_layer_activation
        outputs["style"] = self.style_layers_activation.copy()
        return outputs

    def compute_loss(self):
        generated_outputs = self.gather_generated_outputs(self.generated_tensor)
        content_cost = compute_content_cost(
            self.content_output.squeeze(),
            generated_outputs["content"].squeeze(),
        )
        total_style_cost = 0
        for idx in self.style_outputs:
            style_output = self.style_outputs[idx]
            generated_output = generated_outputs["style"][idx]
            style_cost = compute_layer_style_cost(
                style_output.squeeze(),
                generated_output.squeeze(),
            )
            total_style_cost += style_cost * self.style_layer_weights[idx]
        total_cost = self.alpha * content_cost + self.beta * total_style_cost
        return total_cost

    def get_generated_img(self):
        tensor = self.denormalizer(self.generated_tensor.detach().cpu()) * 255
        img_arr = tensor.numpy().squeeze().clip(0, 255)
        img_arr = np.moveaxis(img_arr, 0, -1).astype(np.uint8)
        return Image.fromarray(img_arr)

    def fit(self, epochs):
        with Progress() as progress:
            task = progress.add_task("Painting...", total=epochs)
            for epoch in range(epochs):
                loss = self.compute_loss()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                msg = "{" + f"loss: {loss.item():.2e}" + "}"
                progress.update(task, description=msg, advance=1)
        return self.get_generated_img()




