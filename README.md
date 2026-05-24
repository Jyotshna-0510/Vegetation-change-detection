# Automated Vegetation Monitoring using Multitemporal Satellite Images

## Overview
This project focuses on automated vegetation monitoring and change detection using multitemporal satellite TIFF imagery and a deep learning-based CF-HGNN-DP model. The system analyzes vegetation variations across different time periods using vegetation change detection techniques and interactive visualizations through a Streamlit web application.

The project helps in monitoring environmental changes, vegetation growth, land-cover transformation, and ecological variations using remote sensing data.

---

## Features
- Multitemporal TIFF satellite image processing
- Automated vegetation monitoring
- Vegetation gain and loss detection
- Deep learning-based prediction using CF-HGNN-DP
- Interactive Streamlit web application
- 2D vegetation change maps
- 3D terrain and vegetation visualization
- Performance metrics visualization
- Real-time vegetation analysis

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
python -m pip install -r requirements.txt
```

### Run Application
```bash
python -m streamlit run app.py
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
- Vegetation preprocessing inputs

---

## Output Visualizations
- 2D Vegetation Change Maps
- 3D Vegetation Visualization
- Performance Metrics
- Vegetation Gain/Loss Analysis
- No Change Percentage Analysis

---

## Screenshots

### Dashboard Interface
![Dashboard](https://github.com/user-attachments/assets/288b64fe-aabe-4571-b46e-4049140b8715)

### TIFF Image Upload
![Upload Interface](https://github.com/user-attachments/assets/c65abaae-0f3a-4e6e-8a37-41a84304e3cf)

### 2D Vegetation Change Detection
![2D Output](https://github.com/user-attachments/assets/c108a60f-0b67-4f2a-9936-4ecba56b6592)

### 3D Vegetation Visualization
![3D Visualization](https://github.com/user-attachments/assets/d1f969ee-4913-48b8-8b2e-9e7fa9dd2b13)

### Performance Metrics
![Performance Metrics](https://github.com/user-attachments/assets/85c6a83f-48c7-4928-966d-8843474f8c8e)

### Vegetation Gain, Loss and No Change Analysis
![Vegetation Analysis](https://github.com/user-attachments/assets/219a8199-4da8-4832-9c42-3ebc06bdcb4d)

---

## Future Improvements
- Support for larger satellite datasets
- Cloud deployment optimization
- Real-time satellite data integration
- Advanced segmentation models
- Multi-class land-cover detection

---

## Author
**Gummadi Jyotshna**  
B.Tech Computer Science Engineering in AIML

---

## GitHub Repository
https://github.com/Jyotshna-0510/Vegetation-change-detection
