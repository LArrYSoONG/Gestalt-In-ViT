# This script is used for generating principle conponens
from datasets import load_dataset
import numpy as np

from transformers import AutoImageProcessor, ViTMAEModel
import torch
from sklearn.decomposition import PCA
import os


def hook(module, input, output):
    outputs_before_dense.append(output[0].cpu())

def remove_all_hooks(module):
    if hasattr(module, '_forward_hooks'):
        module._forward_hooks.clear()



if __name__ == '__main__':
    # load dataset
    ds = load_dataset("apple/flair")

    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # No mask needed, mask_ratio = 0
    processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.config.mask_ratio = 0
    model.config.output_attentions = True
    model.to("cuda")

    # Set up hook
    for i, layer in enumerate(model.encoder.layer):
        remove_all_hooks(layer.attention.attention)
    outputs_before_dense = []
    for i, layer in enumerate(model.encoder.layer):
        layer.attention.attention.register_forward_hook(hook)


    # Extract image features
    for idx, data in enumerate(ds['train']):
        image = data['image']
        image = np.array(image)[:, :, 0:3]
        inputs = processor(images=image, return_tensors="pt")

        # Run model for this image
        with torch.no_grad():
            outputs = model(inputs.pixel_values.to("cuda"), output_attentions=True, output_hidden_states=True, return_dict=True)
        if idx%100==0:
            print(idx)
        if idx == 1000:
            break
    
    # Reshape to (head_num, image_num, feature_dim)
    target = np.array(outputs_before_dense)
    target = target.reshape(-1, 12, 197, 12, 64)
    target = target.transpose(0, 1, 3, 2, 4)
    target = target.reshape(-1, 144, 197, 64)
    target = target.transpose(1, 0, 2, 3)
    target = target.reshape(144, -1, 64)

    # Calculate Principle component
    pcas = []
    for head in target:
        pca = PCA(n_components=10)
        pca.fit(head)
        pcas.append(pca)

    # Save components
    save_dir = './pca_components/'
    os.makedirs(save_dir, exist_ok=True)

    for i, pca in enumerate(pcas):
        np.savez(f'{save_dir}pca_{i}.npz', components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_)
