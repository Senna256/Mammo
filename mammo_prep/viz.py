import numpy as np
import matplotlib.pyplot as plt

def show(img, title=""):
    """
    Shows an image on grayscale whith title.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_comparison(images, titles=None, figsize=(15,5)):
    """
    Shows a bunch of imeges in a row to compare the diferent steps of the pipeline.

    Example:
        plot_comparison([original, normalized, clahe], 
                        titles=["Original", "Normalized", "CLAHE"])
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n==1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles or [""] * n):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()