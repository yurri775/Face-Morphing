# Guide d'Intégration du Dataset Olivetti dans votre Projet de Morphing

## 🚀 Démarrage Rapide

### 1. Installation

```bash
pip install kagglehub numpy opencv-python matplotlib
```

### 2. Utilisation Basique

```python
from face_morphing_wrapper import FaceMorphingWrapper

# Créer le wrapper
wrapper = FaceMorphingWrapper("path/to/metadata.json")

# Obtenir une paire pour le morphing
face1, face2, info = wrapper.get_face_pair_for_morphing()

# Vos algorithmes de morphing ici
result = your_morphing_algorithm(face1, face2)
```

## 📊 Informations sur le Dataset

- **Total d'images**: 2000 (400 originales + 1600 augmentées)
- **Résolution**: 64x64 pixels
- **Format**: Niveaux de gris (automatiquement converti en RGB)
- **Personnes**: 40 (50 images par personne)
- **Augmentations**: Retournements, rotations, bruit, recadrage

## 🔧 Intégration avec vos Algorithmes

### Pour des algorithmes nécessitant des images spécifiques:

```python
# Images en niveaux de gris
face1_gray, face2_gray = wrapper.preprocess_for_your_morphing_algorithm(
    face1, face2, target_size=(256, 256), grayscale=True
)

# Images couleur redimensionnées
face1_color, face2_color = wrapper.preprocess_for_your_morphing_algorithm(
    face1, face2, target_size=(512, 512), grayscale=False
)
```

### Pour des lots d'images:

```python
# Créer un dataset de morphing
morphing_data = wrapper.create_morphing_sequence_data(num_pairs=20)

# Accéder aux paires
for (face1, face2), info in zip(morphing_data['pairs'], morphing_data['metadata']):
    # Votre traitement ici
    pass
```

## 🎯 Cas d'Usage Recommandés

1. **Entraînement de modèles**: Utilisez les 2000 images pour l'entraînement
2. **Test de robustesse**: Les augmentations testent la robustesse de vos algorithmes
3. **Validation croisée**: Séparez par personnes pour éviter le data leakage
4. **Morphing progressif**: Utilisez les variations d'une même personne

## 🐛 Résolution des Problèmes

### Erreur OpenCV avec float64:

✅ **Corrigé** - Le code convertit automatiquement en float32

### Images trop petites (64x64):

```python
# Redimensionner automatiquement
face1, face2 = wrapper.preprocess_for_your_morphing_algorithm(
    face1, face2, target_size=(256, 256)
)
```

### Problèmes de normalisation:

```python
# Les images sont automatiquement normalisées dans [0, 1]
# Pour [0, 255]: multiplier par 255 et convertir en uint8
face_uint8 = (face * 255).astype(np.uint8)
```

## 📁 Export pour Outils Externes

```python
# Exporter vers un dossier
export_path = wrapper.export_for_external_morphing_tool("my_morphing_data")

# Structure créée:
# my_morphing_data/
# ├── pair_0_source_person_1.png
# ├── pair_0_target_person_15.png
# ├── pair_1_source_person_3.png
# └── morphing_metadata.json
```

## 🔄 Pipeline Complet Recommandé

```python
# 1. Initialiser
wrapper = FaceMorphingWrapper(metadata_file)

# 2. Créer un dataset personnalisé
morphing_data = wrapper.create_morphing_sequence_data(50)

# 3. Pour chaque paire
for i, ((face1, face2), info) in enumerate(zip(morphing_data['pairs'], morphing_data['metadata'])):
    # 4. Préprocesser
    proc_face1, proc_face2 = wrapper.preprocess_for_your_morphing_algorithm(face1, face2)

    # 5. Appliquer votre algorithme
    morphed_sequence = your_morphing_algorithm(proc_face1, proc_face2)

    # 6. Sauvegarder les résultats
    save_morphing_result(morphed_sequence, f"result_{i}")
```

## ✅ Validation

Utilisez `debug_dataset.py` pour vérifier l'intégration:

```bash
python debug_dataset.py
```
