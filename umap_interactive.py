import numpy as np
import umap
import os
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO

# Load embeddings
embeddings = np.load("image_embeddings.npy")

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

# Image folder
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"
filenames = os.listdir(image_folder)

