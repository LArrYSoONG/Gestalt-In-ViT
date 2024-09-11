# Useful functions
from transformers import AutoImageProcessor, ViTMAEModel
import torch
import numpy as np

def load_model():
    processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.config.mask_ratio = 0
    model.config.output_attentions = True
    model.to("cuda")
    return processor, model


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

