
This project was made to experiment urban green area segmentation from UAV imagery using pseudo-labeling and deep learning. 
It was made a comparison between a classical SRM baseline and an edge-aware pipeline (K-Means + Sobel), with ~400 generated pseudo-labels used to train a U-Net for binary vegetation mapping.


# Urban Green Area Segmentation from UAV Imagery

Pipeline for **urban vegetation segmentation** using **pseudo-labeling** and **deep learning**.  
We compare a **classical unsupervised baseline** (SRM via Felzenszwalb) with an **edge-aware pseudo-labeling strategy (v2.6)** that combines **K-Means** and **Sobel**.  
The pseudo-labels supervise a **U-Net** trained for binary vegetation masks on the Semantic Drone Dataset.

---

## Highlights

- ~**400** pseudo-labels generated (v2.6) from RGB images only  
- **U-Net** trained without manual pixel annotations  
- Best **validation IoU ≈ 0.65** (epoch 16) after **25 epochs**  
- Runs on **2× NVIDIA T4** with **DataParallel + AMP**

---

## Dataset

**Semantic Drone Dataset** (RGB aerial images with urban scenes).  
You can find it in `https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset`
Folder structure used in this project:

- `dataset/semantic_drone_dataset/original_images/` → RGB `.jpg`
- `dataset/semantic_drone_dataset/label_images_semantic/` → semantic `.png` (not used for training)
- `RGB_color_image_masks/RGB_color_image_masks/` → color-coded masks (reference)
- **Pseudo-labels v2.6** stored as binary `.png` (`{0,255}`), one-to-one with images:
  - Example: `001.jpg` ↔ `001.png`

---

## Methodology

### 1) Baseline: SRM (Felzenszwalb)

Unsupervised over-segmentation via `skimage.segmentation.felzenszwalb`.

- **Parameters**
  - `scale=500` → coarseness (higher = larger regions)
  - `sigma=0.8` → Gaussian pre-smoothing (reduces texture noise)
  - `min_size=200` → minimum region size (removes tiny fragments)
- **Vegetation tag** via green heuristics (ExG/NGRDI) computed per-region.

> SRM is used as a **reference baseline** and region proposal mechanism.

### 2) Edge-Aware Pseudo-Labeling 

**Per-pixel 4D feature vector**:
- **ExG**: \(2G - R - B\)  
- **NGRDI**: \((G - R) / (G + R)\)  
- **Hue** (HSV)  
- **Sobel magnitude** (edge strength)

**Steps**
1. **K-Means (k=2)** in \([ExG, NGRDI, Hue, Sobel]\)
2. Select vegetation cluster by **highest** \(\text{mean(NGRDI)} + \text{mean(Sobel)}\)
3. **Post-process**: morphological closing + removal of small components

Result: ~**400** refined pseudo-labels (v2.6).

### 3) U-Net Segmentation

- **Architecture**: encoder–decoder, skip connections
- **Loss**: BCE + Dice
- **Optimizer**: Adam (lr = 1e-3)
- **Batch size**: 4
- **Input size**: 512×512 (with flips + mild HSV jitter)
- **Training**: 25 epochs, **2× T4** (DataParallel), **AMP** enabled

---

## Results

### Training (25 epochs)

- **Best Val IoU**: **0.6504** at **epoch 16**
- Convergence within ~15 epochs, stable afterward

### Quantitative comparison

| Method                   | IoU  | Dice | Precision | Recall |
|-------------------------|:----:|:----:|:---------:|:------:|
| SRM + Green Rule        | 0.42 | 0.59 |   0.61    |  0.57  |
| Pseudo-labels v2.6      | 0.53 | 0.69 |   0.72    |  0.67  |
| **U-Net (25 epochs)**   | **0.65** | **0.75** | **0.77** | **0.73** |

**Takeaways**
- SRM is a useful unsupervised baseline but tends to **over-segment** vegetation and miss coverage.
- K-Means + Sobel masks are **sharper** and more **boundary-aware**.
- U-Net trained on K-Means + Sobel **outperforms** both SRM and raw pseudo-labels.

---

## Requirements

- Python 3.9+
- PyTorch (CUDA recommended)
- scikit-image
- scikit-learn
- OpenCV-Python
- NumPy, Matplotlib

##  How to Run

### Clone
```bash
git clone https://github.com/x-Lele-x/urban-green-areas-segmentation.git
cd urban-green-segmentation
```

### Open the notebook
```bash
jupyter notebook srm-kmeans-sobel-comparison.ipynb
```

### Execute cells to:
- Generate **SRM** and **v2.6 pseudo-labels**  
- Train **U-Net** on v2.6  
- Evaluate metrics and visualize predictions  

### Paths in the notebook expect:
- **Images**: `.../original_images/`  
- **Pseudo-labels v2.6**: `.../pseudolabels_sobel/` (binary `.png`)  

---

## Notes on Felzenszwalb Parameters
- **scale** → controls merge tolerance. Higher → fewer, larger regions  
- **sigma** → Gaussian blur before graph building. Higher → smoother, less texture-driven fragmentation  
- **min_size** → removes tiny regions. Higher → suppresses small artifacts, but may merge thin vegetation  

---

## Limitations & Future Work

**Limitations**
- Sensitive to shadows and artificial green (painted surfaces)  
- Training set limited (~400 images)  
- RGB-only; no NIR  

**Future Improvements**
- Iterative pseudo-label refinement (self-training / teacher–student)  
- Larger, more diverse UAV datasets  
- Advanced models (DeepLabV3+, Swin-UNet)  
- Multimodal inputs (RGB + NIR)  

---

## 📜 Citation
```bibtex
@misc{giuffrida_urban_green_areas_segmentation_2025,
  author = {Giuffrida, Eleonora},
  title  = {Urban Green Areas Segmentation: comparison between
Sobel-Guided K-Means and SRM strategy},
  year   = {2025},
  note   = {GitHub repository}
}
```


## Author  

**Eleonora Giuffrida**  
*M.Sc. in Computer Science*  
