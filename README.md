# Robust and accurate diagnosis of infectious skin diseases from histopathology images by integrating deep learning and explainable AI

# Authors: 
Kpetchehoue Merveille Santi Zinsou, Habone Ahmed Mahamoud, Abdou Magib Gaye, Idy Diop, Maodo Ndiaye, Doudou Sow, Cheikh Talibouya Diop, and Dmitry Korkin 

# Abstract:
Accurate diagnosis of infectious skin diseases remains a major challenge, particularly for neglected tropical diseases such as mycetoma, where precise pathogen identification is crucial for effective treatment. Histopathology imaging is the diagnostic gold standard, involving examination of tissue biopsies to identify characteristic inflammatory patterns, cellular changes, or microbial pathogens. However, its analysis is often limited by variability in tissue sampling and staining, subjective interpretation, inter-observer differences, and the absence of visible microbial grains in early disease stages. To elevate these challenges, we develop the Skin INfectious Diseases Intelligent (SINDI) framework, an integrated machine learning pipeline combining shallow learning, deep learning, stain normalization, and explainable AI to automate and enhance diagnostic accuracy from histopathology images. The SINDI framework is designed to systematically tackle increasingly complex tasks in diagnostics, including (1) disease phenotype classification, (2) pathogen species identification,  (3) classification of grain-free images lacking visible microbial structures, (4) semantic segmentation of pathological features, and (5) explainable AI-driven interpretable decision support. Leveraging a comprehensive dataset of 1,324 histopathology images representing four predominant mycetoma pathogens that are curated by expert pathologists, alongside 7,000 healthy skin tissue images, SINDI demonstrated near-perfect accuracy in binary and multi-class classification tasks, particularly when employing Macenko stain normalization and domain-specific features. Remarkably, SINDI achieved high accuracy on images with masked grain regions and even on grain-free images, which are considered diagnostically intractable by human experts. Semantic segmentation models accurately delineated phenotype-related regions, while explainable AI methods provided transparent and clinically relevant interpretability of model decisions. Our results indicate that diagnostically relevant information is distributed beyond visible lesion areas, challenging traditional pathology paradigms. The SINDI framework thus represents a significant advance in automated infectious skin disease diagnostics, offering robust, interpretable, and scalable decision-support tools adaptable to diverse clinical settings. 

# SINDI-framework Step-by-Step Tutorial

## Repository Structure
```
SINDI-framework/
├── Explainable AI/
│   └── GradcamPlusPlus-NET++-Efficientnet-b3-XAI.ipynb  # XAI on Segmentation Model
├── Preprocessing/
│   ├── Grain-free image generation.ipynb                  # Creates background-only dataset
│   ├── Heatmaps_Grad_CAM_&Grad_CAM++&_Saliency_gradient_based.ipynb  # Generates XAI heatmaps
│   ├── Split_image_into_ROI_and_Non_ROI.ipynb             # Creates synthetic images for training
│   └── stain_normalization.ipynb                          # Compares Reinhard, Macenko, Vahadane
├── classification/
│   ├── MONAI-DenseNet121/
│   │   └── monai_Macenko_ROI_mask2.ipynb                  # Deep Learning Classifier (Core)
│   ├── domain-independent/
│   │   ├── Logistic regression- MACENKO.ipynb             # Classical ML with Img2Vec features
│   │   └── RF - DT- MACENKO.ipynb                         # Classical ML (Random Forest/Decision Tree)
│   └── domain-specific/
│       └── MACENKO SET.ipynb                              # Domain-specific feature extraction (Color, Texture, etc.)
├── semantic-segmentation/
│   └── UNet++_EfficientnetB3.ipynb                      # Advanced Semantic Segmentation model
├── data/
│   ├── raw/        # Original images
│   └── processed/  # Preprocessed images and masks
├── results/
│   ├── models/      # Trained models
│   └── heatmaps/    # Grad-CAM/Saliency outputs
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 1. Clone the Repository
Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/Merveille-Santi/SINDI-framework.git
cd SINDI-framework
```

This downloads all the files to your computer.

---

## 2. Create a Python Environment
To avoid conflicts, create a virtual environment:

```bash
# Windows
python -m venv sindi_env
sindi_env\Scripts\activate

# Mac/Linux
python3 -m venv sindi_env
source sindi_env/bin/activate
```

---

## 3. Install Dependencies
Install the required packages:

```bash
# Core dependencies
pip install -r requirements.txt

# Optional development dependencies (for Jupyter notebooks)
pip install -r requirements-dev.txt
```

This installs all the Python libraries needed for preprocessing, training, segmentation, and XAI.

---

## 4. Prepare Your Data
1. Create a folder structure:

```
SINDI-framework/data/
├── raw/        # Original images go here
├── processed/  # Preprocessed images will be saved here
```

2. Place your histopathology images in `data/raw/`.

---

## 5. Manual Annotation for Semantic Segmentation
- Install **Label Studio (version 1.13.1)** for manual annotations.
- Open Label Studio and create a project.
- Import your raw images from `data/raw/`.
- Annotate the regions of interest (grains, tissue structures, etc.) to generate **ground truth masks**.
- Export the annotated masks (image mask format .png / .jpg) and save them in `data/processed/masks/`.

> These masks will serve as **ground truth** for training the segmentation models.

---

## 6. Run Preprocessing
1. **Stain Normalization**: Run `Preprocessing/stain_normalization.ipynb` to normalize colors across your images.
2. **ROI/Non-ROI Split**: Run `Preprocessing/Split_image_into_ROI_and_Non_ROI.ipynb` to create synthetic training images.
3. **Grain-free Images**: Run `Preprocessing/Grain-free image generation.ipynb` if you need background-only images.

Outputs will be saved in `data/processed/`.

---

## 7. Train or Run Classification Models
- **Deep Learning (MONAI DenseNet121)**: `classification/MONAI-DenseNet121/monai_Macenko_ROI_mask2.ipynb`
- **Classical ML (domain-independent features)**:
  - Logistic Regression: `classification/domain-independent/Logistic regression- MACENKO.ipynb`
  - Random Forest / Decision Tree: `classification/domain-independent/RF - DT- MACENKO.ipynb`
- **Domain-specific features**: `classification/domain-specific/MACENKO SET.ipynb`

Each notebook contains instructions to load your processed images, train a model, and evaluate results.

---

## 8. Generate Classification Explainability Maps
- Run `Preprocessing/Heatmaps_Grad_CAM_&Grad_CAM++&_Saliency_gradient_based.ipynb`
- Input: Trained classification model + processed images
- Output: Grad-CAM, Grad-CAM++, and Saliency heatmaps saved in `results/heatmaps/`.

> Use this notebook **after training your classification models** to understand which regions influenced predictions.

---

## 9. Semantic Segmentation
- Run `semantic-segmentation/UNet++_EfficientnetB3.ipynb`
- Input: Preprocessed images (`data/processed/`) and ground truth masks (`data/processed/masks/`)
- Output: Segmentation masks saved in `results/`.

---

## 10. Generate Segmentation Explainable AI Maps
- Run `Explainable AI/GradcamPlusPlus-NET++-Efficientnet-b3-XAI.ipynb`
- Input: Trained segmentation/classification model
- Output: Grad-CAM / Grad-CAM++ / Saliency heatmaps saved in `results/heatmaps/`.

---

## 11. Check Results
- Preprocessed images → `data/processed/`
- Ground truth masks → `data/processed/masks/`
- Trained models → `results/models/`
- Heatmaps / Segmentation masks → `results/heatmaps/`

Open any image using `matplotlib` or your favorite image viewer.

---

## 12. Optional: Run All in Script Mode
Once you are confident, you can convert notebooks to `.py` scripts for automation:

```bash
# Example
python scripts/preprocess.py
python scripts/train_classifier.py
python scripts/train_segmentation.py
python scripts/generate_xai_maps.py
```

This allows non-interactive, one-click runs.

---

## Tips for Zero Programming Users
1. Follow the **notebook order**: Preprocessing → Classification → Classification XAI → Semantic Segmentation → Segmentation XAI.
2. Always activate your environment (`sindi_env`) before running notebooks.
3. Keep raw data separate from processed images to avoid overwriting.
4. Use the output folders (`results/`) to visualize trained models and heatmaps.

