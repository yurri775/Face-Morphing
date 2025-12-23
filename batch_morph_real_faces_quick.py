import os
import sys
import subprocess
from pathlib import Path
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
images_dir = 'images'
output_dir = 'results/morph_all'
os.makedirs(output_dir, exist_ok=True)

# Lister toutes les images
image_files = sorted([f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# OPTIMISATION: Traiter TOUTES les images
# (pour test rapide - vous pouvez augmenter ce nombre)
limit = len(image_files) - 1

print('Real Faces Morphing - FULL BATCH (All 500 images)')
print(f'Total images available: {len(image_files)}')
print(f'Processing: {limit} pairs')
print(f'Output: {output_dir}/')
print()

success_count = 0
failed_count = 0
failed_pairs = []

for i in range(limit):
    img1_name = image_files[i]
    img2_name = image_files[i + 1]
    
    img1_path = os.path.join(images_dir, img1_name)
    img2_path = os.path.join(images_dir, img2_name)
    
    # Nom de la sortie: pair_000_001.mp4
    output_name = f'pair_{i:03d}_{i+1:03d}.mp4'
    output_path = os.path.join(output_dir, output_name)
    
    # Ignorer si existe déjà
    if os.path.exists(output_path):
        print(f'[{i+1:3d}/{limit}] SKIP {output_name} (exists)')
        success_count += 1
        continue
    
    print(f'[{i+1:3d}/{limit}] Processing: {img1_name} -> {img2_name}', end=' ... ')
    sys.stdout.flush()
    
    try:
        # Executer le morphing avec timeout genereuse
        cmd = [
            'python', 'code/__init__.py',
            '--img1', img1_path,
            '--img2', img2_path,
            '--output', output_path,
            '--duration', '2',
            '--frame', '15'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0 and os.path.exists(output_path):
            success_count += 1
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f'OK ({file_size:.1f}MB)')
        else:
            failed_count += 1
            failed_pairs.append(output_name)
            print(f'FAILED')
            if result.stderr:
                print(f'         Error: {result.stderr[:200]}')
    
    except subprocess.TimeoutExpired:
        failed_count += 1
        failed_pairs.append(output_name)
        print(f'TIMEOUT')
    except Exception as e:
        failed_count += 1
        failed_pairs.append(output_name)
        print(f'ERROR: {str(e)[:100]}')

print()
print('=' * 60)
print('SUMMARY')
print(f'   Success: {success_count}/{limit}')
print(f'   Failed:  {failed_count}/{limit}')
print(f'   Output: {output_dir}/')

if failed_pairs:
    print(f'\nFailed pairs:')
    for pair in failed_pairs[:10]:
        print(f'   - {pair}')

print('=' * 60)
print('\nNOTE: Processing all 500 images.')
print(f'Total pairs to process: {limit}')

