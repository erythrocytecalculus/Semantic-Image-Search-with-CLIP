import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# Load embeddings
embeddings = np.load("image_embeddings.npy")

# Run UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

# Image folder
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"
filenames = os.listdir(image_folder)