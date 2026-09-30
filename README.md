# brain-tumor-multimodal-cnn

# Multimodal Brain Tumor Classification via Fractional Residual CNNs
This project uses the following dataset: Murtozalikhon. (2023). Brain Tumor Multimodal Image (CT and MRI). Kaggle.
https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri
This repository implements a dual-branch Convolutional Neural Network designed to classify brain tumors (Healthy vs. Tumor) by fusing multimodal medical imaging data (MRI and CT scans). The architecture leverages Transfer Learning (EfficientNetB0) enhanced by custom Fractional Calculus convolutional layers.

## Key Features & Architecture
* **Multimodal Fusion:** Processes aligned MRI and CT images simultaneously through independent feature extraction branches before concatenating them for final classification.
* **Custom Fractional Layer:** Implements a custom Keras layer (`FractionalConv2D`) based on fractional derivative masks (alpha=0.6) to highlight edge anomalies and tumor boundaries prior to deep feature extraction.
* **Residual Connections:** Uses Additive residual connections linking the raw input with the fractional output to preserve spatial morphology while enhancing pathological features.
* **Transfer Learning:** Utilizes pre-trained `EfficientNetB0` backbones for robust, high-level feature extraction.

## Performance & Results
The multimodal approach significantly outperforms single-modality models (MRI-only or CT-only). By combining the fractional residual filtering with the multimodal fusion, the model achieves state-of-the-art results:

* **Final Accuracy (Validation):** 99.65%
* **AUC ROC:** 0.9999
* **Loss:** 0.0184

<img width="1464" height="690" alt="matricemultimodal" src="https://github.com/user-attachments/assets/85fd707e-96fd-43c8-9875-385936016085" />
<img width="790" height="590" alt="rocmultimodal" src="https://github.com/user-attachments/assets/1fa92d30-5682-4136-ba84-617dbcb7cc77" />

## Project Structure
The pipeline evaluates and compares multiple architectures:
1. **Unimodal Baselines:** Standard EfficientNetB0 on MRI and CT independently.
2. **Fractional Unimodals:** Custom fractional layers applied to independent modalities.
3. **Multimodal Baseline:** Standard fusion of MRI and CT features.
4. **Fractional Multimodal (Proposed):** The full pipeline combining custom fractional filters, residual connections, and multimodal fusion.

 Usage
**1. Install dependencies:**
`pip install -r requirements.txt`

**2. Data Setup:**
The project uses the "Brain Tumor Multimodal Image (CT and MRI)" dataset via `kagglehub`. The script automatically handles downloading and train/test splitting.
