import numpy as np
from util import load_model, show_head
from PIL import Image, ImageTk, ImageDraw
from matplotlib.colors import TwoSlopeNorm
from matplotlib import pyplot as plt
import cv2
from sklearn.decomposition import PCA

outputs_before_dense = []
def hook(module, input, output):
    outputs_before_dense.append(output[0].cpu())

def remove_all_hooks(module):
    if hasattr(module, '_forward_hooks'):
        module._forward_hooks.clear()

# Load model and dataset
processor, model = load_model()
for i, layer in enumerate(model.encoder.layer):
    layer.attention.attention.register_forward_hook(hook)


# Load pcas
save_dir = './pca_components/'
pcas = []
for i in range(144):
    # Reconstructing the PCA object
    loaded_data = np.load(f'{save_dir}pca_'+str(i)+'.npz')
    pca_reconstructed = PCA(n_components=10)
    pca_reconstructed.components_ = loaded_data['components']
    pca_reconstructed.mean_ = loaded_data['mean']
    pca_reconstructed.explained_variance_ = loaded_data['explained_variance']
    pcas.append(pca_reconstructed)


image = Image.open('fg49.png')
image = np.array(image)[:, :, 0:3]
inputs = processor(images=image, return_tensors="pt")
input_image = np.array(inputs.pixel_values[0].permute(1, 2, 0).cpu())
input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

image_224 = Image.fromarray((input_image * 255).astype(np.uint8))
# image_14 = Image.fromarray((norm_image[:, :, 0] * 255).astype(np.uint8), mode='L').resize((224, 224), Image.NEAREST)

reduced_head, final_head = show_head(image, processor, model, outputs_before_dense, pcas, 140)

norm = TwoSlopeNorm(vmin=np.min(reduced_head), vcenter=0, vmax=np.max(reduced_head))

def highlight_area(event):
    x, y = int(event.xdata // 16), int(event.ydata // 16)

    highlighted_image = image_224.copy()
    draw = ImageDraw.Draw(highlighted_image)
    top_left = (x * 16, y * 16)
    bottom_right = (top_left[0] + 16, top_left[1] + 16)
    draw.rectangle([top_left, bottom_right], outline="red", width=2)

    ax1.imshow(highlighted_image)
    fig.canvas.draw()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.imshow(image_224)
ax1.set_title("224x224 Image")
ax1.axis('off')

# for other two images
# ax2.imshow(image_14, cmap='gray')
# ax2.set_title("14x14 Image (Scaled)")
# ax2.axis('off')

plt.imshow(cv2.resize(final_head[:, :], (224, 224), interpolation=cv2.INTER_NEAREST), cmap='coolwarm', norm=norm)
ax2.set_title("14x14 Image (Scaled)")
ax2.axis('off')

fig.canvas.mpl_connect('button_press_event', highlight_area)

plt.tight_layout()
plt.show()