import kagglehub
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt

class OlivettiDatasetLoader:
    """Gestionnaire complet pour le dataset Olivetti Faces augmenté"""
    
    def __init__(self, metadata_file: Optional[str] = None, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_path = None
        self.faces = None
        self.labels = None
        self.metadata = None
        
        # Charger les métadonnées si disponibles
        if metadata_file and Path(metadata_file).exists():
            self.load_metadata(metadata_file)
    
    def load_metadata(self, metadata_file: str) -> None:
        """Charge les métadonnées JSON du dataset"""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"✓ Métadonnées chargées depuis: {metadata_file}")
    
    def download_dataset(self) -> str:
        """Télécharge le dataset depuis Kaggle avec kagglehub"""
        print("=== Téléchargement du Dataset Olivetti Augmenté ===")
        print("Utilisation de kagglehub pour télécharger...")
        
        # Code kagglehub intégré
        path = kagglehub.dataset_download("martininf1n1ty/olivetti-faces-augmented-dataset")
        
        self.dataset_path = Path(path)
        print(f"✓ Dataset téléchargé dans: {path}")
        
        # Afficher les informations du dataset si métadonnées disponibles
        if self.metadata:
            self._print_dataset_info()
        
        return path
    
    def _print_dataset_info(self) -> None:
        """Affiche les informations du dataset à partir des métadonnées"""
        if not self.metadata:
            return
            
        print(f"\n=== Informations du Dataset ===")
        print(f"📊 Nom: {self.metadata.get('name', 'N/A')}")
        print(f"👨‍💻 Créateur: {self.metadata.get('creator', {}).get('name', 'N/A')}")
        print(f"📝 Version: {self.metadata.get('version', 'N/A')}")
        print(f"📅 Date de publication: {self.metadata.get('datePublished', 'N/A')}")
        print(f"📄 Licence: {self.metadata.get('license', {}).get('name', 'N/A')}")
        
        # Extraire les détails de la description
        desc = self.metadata.get('description', '')
        if '2000' in desc:
            print(f"🖼️  Total d'images: 2000 (400 originales + 1600 augmentées)")
            print(f"👥 Personnes: 40 (50 images par personne)")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Charge les fichiers .npy du dataset"""
        if self.dataset_path is None:
            self.download_dataset()
            
        faces_path = self.dataset_path / "augmented_faces.npy"
        labels_path = self.dataset_path / "augmented_labels.npy"
        
        if not faces_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Fichiers non trouvés dans: {self.dataset_path}")
        
        print("\n=== Chargement des Données ===")
        self.faces = np.load(str(faces_path))
        self.labels = np.load(str(labels_path))
        
        # Validation des données
        self._validate_data()
        
        return self.faces, self.labels
    
    def _validate_data(self) -> None:
        """Valide les données chargées contre les métadonnées"""
        print(f"✓ Images chargées: {self.faces.shape}")
        print(f"✓ Labels chargés: {self.labels.shape}")
        print(f"✓ Type de données: {self.faces.dtype}")
        print(f"✓ Plage de valeurs: [{self.faces.min():.3f}, {self.faces.max():.3f}]")
        
        # Vérifications contre métadonnées
        if self.metadata:
            expected_total = 2000
            if len(self.faces) == expected_total:
                print(f"✅ Nombre d'images correct: {expected_total}")
            else:
                print(f"⚠️  Nombre d'images inattendu: {len(self.faces)} vs {expected_total}")
        
        # Statistiques des personnes
        unique_persons = len(np.unique(self.labels))
        print(f"✓ Personnes uniques détectées: {unique_persons}")
        
        # Images par personne
        person_counts = np.bincount(self.labels)
        print(f"✓ Images par personne: min={person_counts.min()}, max={person_counts.max()}, moyenne={person_counts.mean():.1f}")
    
    def get_face_by_person(self, person_id: int, max_images: Optional[int] = None) -> np.ndarray:
        """Récupère les images d'une personne spécifique"""
        if self.faces is None or self.labels is None:
            self.load_data()
            
        mask = self.labels == person_id
        person_faces = self.faces[mask]
        
        if max_images:
            person_faces = person_faces[:max_images]
            
        return person_faces
    
    def get_random_faces(self, count: int = 2) -> List[np.ndarray]:
        """Récupère des visages aléatoires"""
        if self.faces is None:
            self.load_data()
            
        indices = np.random.choice(len(self.faces), count, replace=False)
        return [self.faces[i] for i in indices]
    
    def preprocess_for_morphing(self, face: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Préprocess une image pour le morphing - VERSION CORRIGÉE"""
        # 1. Convertir en float32 pour éviter les problèmes OpenCV avec float64
        face = face.astype(np.float32)
        
        # 2. Normaliser les valeurs dans [0, 1]
        if face.max() > 1.0:
            face = face / 255.0
        
        # 3. S'assurer que les valeurs sont strictement dans [0, 1]
        face = np.clip(face, 0.0, 1.0)
        
        # 4. Redimensionner si nécessaire
        if face.shape[:2] != target_size:
            face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
            
        # 5. Convertir en RGB sans utiliser cv2.cvtColor (problématique avec float64)
        if len(face.shape) == 2:
            # Pour les images en niveaux de gris, créer 3 canaux identiques
            face = np.stack([face, face, face], axis=-1)
        elif len(face.shape) == 3 and face.shape[2] == 1:
            # Si c'est (H, W, 1), répliquer sur 3 canaux
            face = np.repeat(face, 3, axis=2)
            
        return face
    
    def display_sample_gallery(self, count: int = 12, persons: Optional[List[int]] = None) -> None:
        """Affiche une galerie d'échantillons"""
        if self.faces is None:
            self.load_data()
            
        if persons:
            # Afficher des échantillons de personnes spécifiques
            faces_to_show = []
            labels_to_show = []
            for person_id in persons[:count]:
                person_faces = self.get_face_by_person(person_id, 1)
                if len(person_faces) > 0:
                    faces_to_show.append(person_faces[0])
                    labels_to_show.append(person_id)
        else:
            # Échantillons aléatoires
            indices = np.random.choice(len(self.faces), min(count, len(self.faces)), replace=False)
            faces_to_show = [self.faces[i] for i in indices]
            labels_to_show = [self.labels[i] for i in indices]
        
        # Affichage avec gestion des types de données
        rows = int(np.ceil(len(faces_to_show) / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, (face, label) in enumerate(zip(faces_to_show, labels_to_show)):
            if i < axes.size:
                ax = axes.flat[i]
                
                # Normaliser et convertir pour affichage
                display_face = face.astype(np.float32)
                if display_face.max() > 1.0:
                    display_face = display_face / 255.0
                
                # S'assurer que les valeurs sont dans [0, 1]
                display_face = np.clip(display_face, 0.0, 1.0)
                    
                ax.imshow(display_face, cmap='gray')
                ax.set_title(f"Personne {label}")
                ax.axis('off')
        
        # Cacher les axes inutilisés
        for i in range(len(faces_to_show), axes.size):
            axes.flat[i].axis('off')
            
        plt.suptitle("Échantillons du Dataset Olivetti Augmenté", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_morphing_pair(self, method: str = "different_persons") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Obtient une paire de visages optimisée pour le morphing"""
        if self.faces is None:
            self.load_data()
        
        if method == "different_persons":
            # Sélectionner deux personnes différentes
            unique_persons = np.unique(self.labels)
            selected_persons = np.random.choice(unique_persons, 2, replace=False)
            
            face1 = self.get_face_by_person(selected_persons[0], 1)[0]
            face2 = self.get_face_by_person(selected_persons[1], 1)[0]
            
            info = {
                'person1_id': int(selected_persons[0]),
                'person2_id': int(selected_persons[1]),
                'method': method,
                'original_shapes': [face1.shape, face2.shape]
            }
        else:  # random
            faces = self.get_random_faces(2)
            face1, face2 = faces[0], faces[1]
            info = {'method': 'random'}
        
        # Préprocesser pour le morphing avec la version corrigée
        processed_face1 = self.preprocess_for_morphing(face1)
        processed_face2 = self.preprocess_for_morphing(face2)
        
        return processed_face1, processed_face2, info
    
    def export_dataset_info(self, output_file: str = "dataset_info.txt") -> None:
        """Exporte les informations du dataset"""
        output_path = self.cache_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== DATASET OLIVETTI FACES AUGMENTÉ ===\n\n")
            
            if self.metadata:
                f.write(f"Nom: {self.metadata.get('name', 'N/A')}\n")
                f.write(f"Créateur: {self.metadata.get('creator', {}).get('name', 'N/A')}\n")
                f.write(f"Version: {self.metadata.get('version', 'N/A')}\n")
                f.write(f"URL: {self.metadata.get('url', 'N/A')}\n")
                f.write(f"Licence: {self.metadata.get('license', {}).get('name', 'N/A')}\n\n")
            
            if self.faces is not None:
                f.write("=== STATISTIQUES ===\n")
                f.write(f"Total d'images: {len(self.faces)}\n")
                f.write(f"Forme des images: {self.faces.shape[1:]}\n")
                f.write(f"Type de données: {self.faces.dtype}\n")
                f.write(f"Personnes uniques: {len(np.unique(self.labels))}\n")
                f.write(f"Plage de valeurs: [{self.faces.min():.3f}, {self.faces.max():.3f}]\n")
        
        print(f"📄 Informations exportées vers: {output_path}")
