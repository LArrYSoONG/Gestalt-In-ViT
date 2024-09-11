from PIL import Image
import numpy as np
from util import add_border_test, calculate_scores, load_model
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# set hook
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

# for an example image, calculate its attention pattern
# and mask for different pixel width border around it
image = Image.open('fg49.png')

all_image_result, masks = add_border_test(image, processor, model, pcas, outputs_before_dense)
# Head with KL divergence over 4 in natural test
verified_head = [65, 66, 67, 71, 80, 89, 90, 95, 104, 109, 110, 111, 114, 124, 126, 132, 134, 138, 140]

# Mean and variance for different areas(figure, ground and edge)
# In each images, with wider border around the image
scores_figure = []
scores_ground = []
scores_edge = []
std_figures = []
std_grounds = []
std_edges = []
for result, mask in zip(all_image_result, masks):
    score_head_figure = []
    score_head_ground = []
    score_head_edge = []
    std_head_figure = []
    std_head_ground = []
    std_head_edge = []
    for head in verified_head:
        score_figure, score_ground, score_edge, std_figure, std_ground, std_edge = calculate_scores(result[head], mask)
        score_head_figure.append(score_figure)
        score_head_ground.append(score_ground)
        score_head_edge.append(score_edge)
        std_head_figure.append(std_figure)
        std_head_ground.append(std_ground)
        std_head_edge.append(std_edge)
    scores_figure.append(score_head_figure)
    scores_ground.append(score_head_ground)
    scores_edge.append(score_head_edge)
    std_figures.append(std_head_figure)
    std_grounds.append(std_head_ground)
    std_edges.append(std_head_edge)

scores_figure = np.array(scores_figure)
scores_ground = np.array(scores_ground)
scores_edge = np.array(scores_edge)
std_figures = np.array(std_figures)
std_grounds = np.array(std_grounds)
std_edges = np.array(std_edges)

# Plot mean and average change for every head with changing 
# border width
for idx, head in enumerate(verified_head):
    plt.title(str(head) + "Average")
    plt.plot(scores_figure[:, idx], color='blue')
    plt.plot(scores_ground[:, idx], color='red')
    plt.plot(scores_edge[:, idx], color='green')
    plt.show()

    plt.title(str(head) + "Std")
    plt.plot(std_figures[:, idx], color='blue')
    plt.plot(std_grounds[:, idx], color='red')
    plt.plot(std_edges[:, idx], color='green')
    plt.show()