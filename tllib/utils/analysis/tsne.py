 
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import sys

# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               target_labels: torch.Tensor, filename: str):
#     """
#     Visualize features from different domains using t-SNE.

#     Args:
#         source_feature (tensor): features from source domain in shape (minibatch, F)
#         target_feature (tensor): features from target domain in shape (minibatch, F)
#         target_labels (tensor): labels for the target features in shape (minibatch,)
#         filename (str): the file name to save t-SNE
#     """
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)

#     # map features to 2-d using TSNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     # domain labels, 1 represents source while 0 represents target
#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

#     # get unique labels from target_labels
#     unique_labels = torch.unique(target_labels)

#     # create a colormap with unique colors for each label
#     num_labels = len(unique_labels)
#     colormap = plt.cm.get_cmap('tab10', num_labels)

#     # create a list of colors for target features based on labels
#     target_colors = [colormap(i) for i in range(num_labels)]

#     # visualize using matplotlib
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

#     # plot target features with corresponding colors based on labels
#     for i, label in enumerate(unique_labels):
#         mask = target_labels == label
#         print(target_labels.shape) #2400
#         print(label)
#         print(mask.shape) #2400
#         mask=mask.cpu().numpy()
#         print(X_tsne.shape)
#         plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=target_colors[i], label=f"Class {label}", s=20)

#     # remove source features by setting their color to transparent
#     plt.scatter(X_tsne[:len(source_feature), 0], X_tsne[:len(source_feature), 1], c=(0, 0, 0, 0), s=20)

#     plt.xticks([])
#     plt.yticks([])
#     # plt.legend()
#     plt.savefig(filename, bbox_inches='tight')

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, target_labels: torch.Tensor, filename_prefix: str):
    """
    Visualize features from the target domain using t-SNE for all classes in one plot.

    Args:
        target_feature (tensor): features from target domain in shape (minibatch, F)
        target_labels (tensor): labels for the target features in shape (minibatch,)
        filename (str): the file name to save the t-SNE plot
    """
    target_feature = target_feature.numpy()

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=50).fit_transform(target_feature)

    # get unique labels from target_labels
    unique_labels = torch.unique(target_labels)

    # create a colormap with unique colors for each label
    num_labels = len(unique_labels)
    colormap = plt.cm.get_cmap('tab10', num_labels)

    # create a list of colors for target features based on labels
    target_colors = [colormap(i) for i in range(num_labels)]

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))

    for i, label in enumerate(unique_labels):
        mask = target_labels == label
        mask=mask.cpu().numpy()
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=target_colors[i], label=f"Class {label}", s=20)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('tsne.png', bbox_inches='tight')
    plt.show()