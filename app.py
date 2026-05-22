import streamlit as st
import os
import json
import io
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Vegetation Change Detection", layout="wide")

# ==========================================
# HELPER FUNCTION
# ==========================================
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# ==========================================
# MODEL (UNCHANGED)
# ==========================================
class CF_HGNN_DP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = CF_HGNN_DP(4)
    model.load_state_dict(torch.load("cf_hgnn_dp_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ==========================================
# LOAD METRICS
# ==========================================
with open("metrics.json", "r") as f:
    saved_metrics = json.load(f)

# ==========================================
# PROCESS FUNCTION (NEW: supports arrays)
# ==========================================
@st.cache_data
def process_arrays(red_t1, nir_t1, red_t2, nir_t2):

    ndvi_t1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + 1e-6)
    ndvi_t2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + 1e-6)
    ndvi_change = ndvi_t2 - ndvi_t1

    ndvi_smooth = gaussian_filter(ndvi_change, sigma=1)
    gradient = sobel(ndvi_change)

    ndvi_uint8 = ((ndvi_change - ndvi_change.min()) /
                  (ndvi_change.max() - ndvi_change.min() + 1e-6) * 255).astype(np.uint8)

    lbp = local_binary_pattern(ndvi_uint8, P=8, R=1, method='uniform')

    features = np.stack([ndvi_change, ndvi_smooth, gradient, lbp], axis=-1)
    features = np.nan_to_num(features)

    segments = slic(features, n_segments=300, compactness=10, channel_axis=-1)

    edges = set()
    h, w = segments.shape
    for i in range(h - 1):
        for j in range(w - 1):
            a = segments[i, j]
            b = segments[i, j + 1]
            c = segments[i + 1, j]

            if a != b:
                edges.add((a, b)); edges.add((b, a))
            if a != c:
                edges.add((a, c)); edges.add((c, a))

    edge_index = torch.tensor(np.array(list(edges)).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)

    node_features = []
    for i in range(segments.max() + 1):
        mask = segments == i
        region = features[mask]

        if len(region) == 0:
            node_features.append(np.zeros(4))
        else:
            node_features.append(np.mean(region, axis=0))

    x = torch.tensor(np.array(node_features), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data, segments, ndvi_change

# ==========================================
# UI
# ==========================================
st.title("🌿 Vegetation Change Detection")
st.caption("CF-HGNN-DP Deep Learning Model with 3D Visualization")

# 🔥 NEW: MODE SELECT
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose Input", ["Dataset", "Upload TIFF"])

base_dataset = "Dataset"
locations = sorted(os.listdir(base_dataset))

# ==========================================
# DATA INPUT
# ==========================================
if mode == "Dataset":
    selected_city = st.sidebar.selectbox("Choose Area", locations)

    def read_tif(path):
        with rasterio.open(path) as src:
            return src.read(1).astype(float)

    city_path = os.path.join(base_dataset, selected_city)

    red_t1 = read_tif(city_path + "/imgs_1_rect/B04.tif")
    nir_t1 = read_tif(city_path + "/imgs_1_rect/B08.tif")
    red_t2 = read_tif(city_path + "/imgs_2_rect/B04.tif")
    nir_t2 = read_tif(city_path + "/imgs_2_rect/B08.tif")

else:
    st.sidebar.write("Upload TIFF files")

    red_t1_file = st.sidebar.file_uploader("T1 Red (B04)", type="tif")
    nir_t1_file = st.sidebar.file_uploader("T1 NIR (B08)", type="tif")
    red_t2_file = st.sidebar.file_uploader("T2 Red (B04)", type="tif")
    nir_t2_file = st.sidebar.file_uploader("T2 NIR (B08)", type="tif")

    if red_t1_file and nir_t1_file and red_t2_file and nir_t2_file:

        def read_upload(file):
            with rasterio.open(file) as src:
                return src.read(1).astype(float)

        red_t1 = read_upload(red_t1_file)
        nir_t1 = read_upload(nir_t1_file)
        red_t2 = read_upload(red_t2_file)
        nir_t2 = read_upload(nir_t2_file)
    else:
        st.warning("Please upload all 4 TIFF files")
        st.stop()

# ==========================================
# RUN ANALYSIS
# ==========================================
if st.sidebar.button("Run Analysis"):

    data, segments, ndvi_change = process_arrays(red_t1, nir_t1, red_t2, nir_t2)

    pred = model(data).argmax(dim=1).detach().numpy()

    final_map = np.zeros_like(segments)
    for i in range(len(pred)):
        final_map[segments == i] = pred[i] - 1

    st.success("Analysis completed")

    col1, col2 = st.columns([2, 1])

    # ======================================
    # MAP + 3D
    # ======================================
    with col1:
        st.subheader("2D Vegetation Change Map")

        cmap = ListedColormap(["red", "lightgray", "green"])
        fig, ax = plt.subplots()
        im = ax.imshow(final_map, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im)

        st.pyplot(fig)

        st.download_button("Download Map", fig_to_png(fig), "map.png")

        st.subheader("3D Surface")
        z = ndvi_change[::5, ::5]

        fig3d = go.Figure(data=[go.Surface(z=z, colorscale="RdYlGn")])
        st.plotly_chart(fig3d, use_container_width=True)

    # ======================================
    # METRICS
    # ======================================
    with col2:
        unique, counts = np.unique(final_map, return_counts=True)
        summary = {-1: 0, 0: 0, 1: 0}
        for u, c in zip(unique, counts):
            summary[u] = c

        total = final_map.size

        st.subheader("Area Percentage")
        st.write(f"Loss: {(summary[-1]/total)*100:.2f}%")
        st.write(f"No Change: {(summary[0]/total)*100:.2f}%")
        st.write(f"Gain: {(summary[1]/total)*100:.2f}%")

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{saved_metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{saved_metrics['precision']*100:.2f}%")
        st.metric("Recall", f"{saved_metrics['recall']*100:.2f}%")
        st.metric("F1 Score", f"{saved_metrics['f1_score']*100:.2f}%")