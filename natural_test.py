from datasets import load_dataset
from util import load_model, generate_all_for_image
import numpy as np
import cv2
from matplotlib import pyplot as plt
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

ds_voc = load_dataset("nateraw/pascal-voc-2012")

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

# Images and masks, mask are transfered into figure, ground and edge, same size as 
# number of patches(14 * 14)
images = []
masks = []
for image, mask in zip(ds_voc['train']['image'], ds_voc['train']['mask']):
    image = np.array(image)
    mask = np.array(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask[mask!=0] = 100
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    for i in range(196):
        mean = mask[i // 14 * 16 : i // 14 * 16 + 16, i % 14 * 16 : i % 14 * 16 + 16].mean()
        if mean != 100 and mean != 0:
            mask[i // 14 * 16 : i // 14 * 16 + 16, i % 14 * 16 : i % 14 * 16 + 16] = 50
    images.append(image)
    mask = cv2.resize(mask, (14, 14), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    masks.append(mask)

# Patch 's PCA projection matrix for every single head
all_heads = []

for image in images:
    final_heads = generate_all_for_image(image, processor, model, pcas, outputs_before_dense)
    all_heads.append(final_heads)


# Calculate mean activation for different area(figure, ground, edge) 
# for each image, and draw distribution for 1500 images. Blue for figure, 
# red for ground and green for edge
figure = []
ground = []
edge = []
KLs = []
verified_head = []

for head_num in range(144):
    for image, mask, final_heads in zip(images, masks, all_heads):
        map = final_heads[head_num]
        map_100 = map[mask==100]
        map_50 = map[mask==50]
        map_0 = map[mask==0]
        if map_100.size == 0 or map_50.size == 0 or map_0.size == 0:
            continue
        # print(1)
        figure.append(map_100.mean())
        edge.append(map_50.mean())
        ground.append(map_0.mean())
        i = i + 1

        # gc.collect()
        # torch.cuda.empty_cache()

        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # plt.imshow(map, cmap='coolwarm', norm=norm)
        # plt.show()
    figure = np.array(figure)
    ground = np.array(ground)
    edge = np.array(edge)

    figure_mean = figure.mean()
    figure_std = figure.std()
    ground_mean = ground.mean()
    ground_std = ground.std()

    KL = np.log(figure_std / ground_std) + (ground_std**2 + (ground_mean - figure_mean) ** 2) / (2 * figure_std ** 2) - 0.5
    KLs.append(KL)
    if KL > 4: 
        verified_head.append(head_num)

    plt.hist(ground, bins=200, range=(-3, 3), edgecolor='red')
    plt.hist(figure, bins=200, range=(-3, 3), edgecolor='blue')
    plt.hist(edge, bins=200, range=(-3, 3), edgecolor='green')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Ground and Figure'+str(head_num))
    plt.savefig('Distribution for head' + str(head_num) + '.png')
    plt.show()
    plt.close()

    i = 0
    figure = []
    ground = []
    edge = []

# plot of KL divergence of figure distribution and ground distribution
# for all heads
plt.plot(KLs)
plt.axhline(y=4, color='red', linestyle='--', linewidth=1)
plt.axvline(x=132, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=120, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=108, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=96, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=84, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=72, color='blue', linestyle='-.', linewidth=1)
plt.axvline(x=60, color='blue', linestyle='-.', linewidth=1)
plt.show()

# KL divergence mean for every layer
KL_ave = []
KLs = np.array(KLs)
for i in range(12):
    KL_ave.append(KLs[i*12:(i+1)*12].mean())
plt.plot(KL_ave)
plt.show()