import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kagglehub

def download_and_prepare_data(img_size=(224, 224), test_split=0.2):
    """
    Downloads the Multimodal Brain Tumor dataset via Kagglehub,
    processes MRI and CT images, and prepares train/test splits.
    """
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("murtozalikhon/brain-tumor-multimodal-image-ct-and-mri")
    
    mri_dir = os.path.join(path, "Dataset", "Brain Tumor MRI images")
    ct_dir = os.path.join(path, "Dataset", "Brain Tumor CT scan Images")
    
    x_mri, x_ct, y = [], [], []
    classes = ['Healthy', 'Tumor']
    
    print("Processing and resizing image pairs...")
    for idx, cls in enumerate(classes):
        mri_cls_p = os.path.join(mri_dir, cls)
        ct_cls_p = os.path.join(ct_dir, cls)
        
        # Sort files to ensure MRI and CT pairs match correctly
        mri_files = sorted(os.listdir(mri_cls_p))
        ct_files = sorted(os.listdir(ct_cls_p))
        
        for mf, cf in zip(mri_files, ct_files):
            img_m = cv2.imread(os.path.join(mri_cls_p, mf))
            img_c = cv2.imread(os.path.join(ct_cls_p, cf))
            
            if img_m is not None and img_c is not None:
                x_mri.append(cv2.resize(img_m, img_size))
                x_ct.append(cv2.resize(img_c, img_size))
                y.append(idx)
                
    X_mri = np.array(x_mri)
    X_ct = np.array(x_ct)
    Y = tf.keras.utils.to_categorical(np.array(y), len(classes))
    
    print("Splitting data into train and test sets...")
    train_mri, test_mri, train_ct, test_ct, train_y, test_y = train_test_split(
        X_mri, X_ct, Y, test_size=test_split, random_state=42
    )
    
    print("Data preparation complete.")
    return train_mri, test_mri, train_ct, test_ct, train_y, test_y
