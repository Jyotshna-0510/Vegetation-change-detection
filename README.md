# Automated Vegetation Monitoring using Multitemporal Satellite Images

## Overview
This project focuses on automated vegetation monitoring and change detection using multitemporal satellite TIFF imagery and a deep learning-based CF-HGNN-DP model. The system analyzes vegetation variations across different time periods using NDVI computation, change detection techniques, and interactive visualizations through a Streamlit web application.

The project helps in monitoring environmental changes, vegetation growth, land-cover transformation, and ecological variations using remote sensing data.

---

## Features
- Multitemporal TIFF satellite image processing
- Automated vegetation monitoring
- NDVI (Normalized Difference Vegetation Index) analysis
- Vegetation gain and loss detection
- Deep learning-based prediction using CF-HGNN-DP
- Interactive Streamlit web application
- 2D vegetation change maps
- 3D terrain and vegetation visualization
- Real-time visualization and analysis

---

## Technologies Used
- Python
- PyTorch
- Torch Geometric
- Streamlit
- Rasterio
- NumPy
- Matplotlib
- Plotly
- Scikit-image
- SciPy

---

## Deep Learning Model
This project uses the **CF-HGNN-DP (Casual feature hybrid graph neural networks with diffusion priors)** model for vegetation monitoring and change detection from multitemporal satellite imagery.

---

## Live Demo
🚀 Streamlit Application:  
https://jyotshna-0510-vegetation-change-detection-app-wzpxhk.streamlit.app/

---

## Installation

### Clone Repository
```bash
git clone https://github.com/Jyotshna-0510/Vegetation-change-detection.git
cd Vegetation-change-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

---

## Project Structure
```text
Vegetation-change-detection/
│
├── app.py
├── requirements.txt
├── metrics.json
├── cf_hgnn_dp_model.pth
├── .gitignore
└── README.md
```

---

## Input Data
- Multitemporal TIFF satellite images
- Remote sensing vegetation datasets
- NDVI-based preprocessing inputs

---

## Output Visualizations
- NDVI Maps
- Vegetation Gain/Loss Maps
- Heatmaps
- 3D Surface Visualization
- Change Detection Results

---

## Future Improvements
- Support for larger satellite datasets
- Cloud deployment optimization
- Real-time satellite data integration
- Advanced segmentation models
- Multi-class land-cover detection

---

## Author
**Jyotshna**  
B.Tech Computer Science Engineering in AIML

---

## GitHub Repository
https://github.com/Jyotshna-0510/Vegetation-change-detection
