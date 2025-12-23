import kagglehub
import os
import shutil

# Télécharger le dataset
path = kagglehub.dataset_download('benin007/human-faces-real-sketch-synthetic')
print('Dataset path:', path)

# Explorer la structure
print('\n=== STRUCTURE DU DATASET ===')
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    
    sub_indent = '  ' * (level + 1)
    print(f'{sub_indent}Dirs: {len(dirs)}, Files: {len(files)}')
    
    if len(files) > 0 and level < 3:
        print(f'{sub_indent}Exemples: {files[:5]}')
    
    if level > 2:
        break

# Compter toutes les images
print('\n=== COMPTAGE DES IMAGES ===')
all_images = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(os.path.join(root, f))

print(f'Total images trouvées: {len(all_images)}')
print(f'\nPremières images:')
for img in all_images[:10]:
    print(f'  {img}')
