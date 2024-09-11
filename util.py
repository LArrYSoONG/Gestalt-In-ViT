# Useful functions
from transformers import AutoImageProcessor, ViTMAEModel
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cv2

# Load model
def load_model():
    processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.config.mask_ratio = 0
    model.config.output_attentions = True
    model.to("cuda")
    return processor, model

# Generate pca projection for every patch onto first PC, resize to
# 14*14 map for every head, 144 head in total for every image
def generate_all_for_image(image, processor, model, pcas, outputs_before_dense):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs.pixel_values.to("cuda"), output_attentions=True, output_hidden_states=True, return_dict=True)
    ids_restore = outputs.ids_restore[0]
    ids_restore = torch.cat([torch.tensor([-1]).to("cuda"), ids_restore])


    heads = outputs_before_dense[-12:]
    heads = np.array(heads)
    heads = heads.squeeze()
    heads = heads.reshape(12, 197, 12, 64)
    heads = heads.transpose(0, 2, 1, 3)
    heads = heads.reshape(-1, 197, 64)

    final_heads = []
    for pca, head in zip(pcas, heads):
        reduced_head = pca.transform(head)
        reduced_head = reduced_head[ids_restore.cpu() + 1, :]
        final_head = reduced_head[1:, :].reshape(14, 14, 10)
        final_head = final_head[:, :, 0]
        # final_head[final_head < 0] = -final_head[final_head < 0] / final_head.min()
        # final_head[final_head > 0] = final_head[final_head > 0] / final_head.max()
        final_heads.append(final_head)
    return final_heads


# Take image and return attention map for head with index head_num
def show_head(image, processor, model, outputs_before_dense, pcas, head_num):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs.pixel_values.to("cuda"), output_attentions=True, output_hidden_states=True, return_dict=True)
    ids_restore = outputs.ids_restore[0]
    ids_restore = torch.cat([torch.tensor([-1]).to("cuda"), ids_restore])


    heads = outputs_before_dense[-12:]
    heads = np.array(heads)
    heads = heads.squeeze()
    heads = heads.reshape(12, 197, 12, 64)
    heads = heads.transpose(0, 2, 1, 3)
    heads = heads.reshape(-1, 197, 64)

    pca = pcas[head_num]
    head = heads[head_num]
    reduced_head = pca.transform(head)
    reduced_head = reduced_head[ids_restore.cpu() + 1, :]
    plt.imshow(image[:, :, 0:3])
    plt.show()

    norm = TwoSlopeNorm(vmin=np.min(reduced_head), vcenter=0, vmax=np.max(reduced_head))
    final_head = reduced_head[1:, :].reshape(14, 14, 10)
    final_head = final_head[:, :, 0]
    plt.imshow(final_head[:, :], cmap='coolwarm', norm=norm)
    plt.colorbar(label='Value Intensity')
    plt.show()

    return reduced_head, final_head

# Take image as input, image normally are artificial images with simply two colors
# Return mask of size (14 * 14) which indicating different areas--figure, ground and
# edge
def extract_mask(image):
    image = np.array(image)
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask[mask == mask[mask.shape[0]//2, mask.shape[1]//2]] = 100
    mask[mask != 100] = 0
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    for i in range(196):
        mean = mask[i // 14 * 16 : i // 14 * 16 + 16, i % 14 * 16 : i % 14 * 16 + 16].mean()
        if mean != 100 and mean != 0:
            mask[i // 14 * 16 : i // 14 * 16 + 16, i % 14 * 16 : i % 14 * 16 + 16] = 50

    mask = cv2.resize(mask, (14, 14), interpolation=cv2.INTER_NEAREST)
    return mask

# Take attention pattern(PC projection) and mask indicating areas-- figure
# ground and edge, return the mean and std deviation for different areas
def calculate_scores(final_head, mask):
    final_100 = final_head[mask==100]
    final_50 = final_head[mask==50]
    final_0 = final_head[mask==0]
    score_figure = final_100.mean()
    score_ground = final_0.mean()
    score_edge = final_50.mean()

    std_figure = final_100.std()
    std_ground = final_0.std()
    std_edge = final_50.std()

    return score_figure, score_ground, score_edge, std_figure, std_ground, std_edge

# Add border around original image, for test here only, we add the border
# of the same color as center pixel
def add_border(image: np.ndarray, border_size: int) -> np.ndarray:
    height, width = image.shape[:2]

    new_height = height + 2 * border_size
    new_width = width + 2 * border_size

    target = image[image.shape[0] // 2, image.shape[1] // 2, :]
    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image (3 channels)
        bordered_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        bordered_image[:, :, 0] = target[0]
        bordered_image[:, :, 1] = target[1]
        bordered_image[:, :, 2] = target[2]
    else:  # Grayscale image (1 channel)
        bordered_image = np.ones((new_height, new_width), dtype=np.uint8) * 255

    # Place the original image in the center of the new image
    bordered_image[border_size:border_size+height, border_size:border_size+width] = image

    return bordered_image

# Take original image as input, add border for 10 pixels everytime
# get the PC projection of patches for all heads, and the mask
def add_border_test(image, processor, model, pcas, outputs_before_dense):
    all_image_result = []
    masks = []
    image = np.array(image)[:, :, 0:3]

    for i in range(50):
        border_image = add_border(image, i * 10)
        mask = extract_mask(border_image)
        final_heads = generate_all_for_image(border_image, processor, model, pcas, outputs_before_dense)
        masks.append(mask)
        all_image_result.append(final_heads)

    return all_image_result, masks