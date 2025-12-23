import kagglehub
import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Télécharger le dataset
path = kagglehub.dataset_download('benin007/human-faces-real-sketch-synthetic')
real_faces_dir = os.path.join(path, 'Human_Faces', 'Real_Faces')

# Dossier de destination
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

print('🔄 Exporting Real Faces to images/ folder...')
print(f'Source: {real_faces_dir}')
print(f'Destination: {output_dir}/')

# Copier toutes les images Real Faces
image_files = sorted([f for f in os.listdir(real_faces_dir) if f.lower().endswith('.png')])
print(f'\n📊 Found {len(image_files)} images')

copied = 0
for idx, img_file in enumerate(image_files, 1):
    src = os.path.join(real_faces_dir, img_file)
    dst = os.path.join(output_dir, img_file)
    
    # Copier directement (les PNG sont déjà au bon format)
    shutil.copy2(src, dst)
    copied += 1
    
    if idx % 50 == 0:
        print(f'  ✅ Copied {idx}/{len(image_files)} images...')

print(f'\n✅ DONE! Exported {copied} images to {output_dir}/')

# Vérifier
exported_images = os.listdir(output_dir)
print(f'\n📁 Verification: {len(exported_images)} files in {output_dir}/')
print(f'Exemples: {exported_images[:5]}')
