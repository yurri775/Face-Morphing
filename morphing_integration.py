import numpy as np
import cv2
from dataset_loader import OlivettiDatasetLoader
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt

class MorphingDatasetHelper:
    """Assistant spécialisé pour l'intégration avec les algorithmes de morphing"""
    
    def __init__(self, metadata_file: str = None):
        self.loader = OlivettiDatasetLoader(metadata_file)
        self.face_cache = {}
    
    def get_optimal_face_pairs(self, count: int = 5) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Récupère plusieurs paires optimales pour le morphing"""
        pairs = []
        
        for _ in range(count):
            face1, face2, info = self.loader.get_morphing_pair("different_persons")
            pairs.append((face1, face2, info))
            
        return pairs
    
    def preprocess_batch(self, faces: List[np.ndarray], 
                        target_size: Tuple[int, int] = (256, 256),
                        normalize: bool = True) -> np.ndarray:
        """Préprocess un lot d'images pour le morphing - VERSION CORRIGÉE"""
        processed = []
        
        for face in faces:
            # 1. Convertir en float32 pour éviter les problèmes
            face = face.astype(np.float32)
            
            # 2. Normaliser si nécessaire
            if normalize and face.max() > 1.0:
                face = face / 255.0
            
            # 3. Clipper les valeurs
            face = np.clip(face, 0.0, 1.0)
            
            # 4. Redimensionner
            if face.shape[:2] != target_size:
                face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
            
            # 5. Gérer les canaux de couleur sans cv2.cvtColor
            if len(face.shape) == 2:
                # Image en niveaux de gris, créer 3 canaux identiques
                face = np.stack([face, face, face], axis=-1)
            elif len(face.shape) == 3 and face.shape[2] == 1:
                # Si c'est (H, W, 1), répliquer sur 3 canaux
                face = np.repeat(face, 3, axis=2)
            
            processed.append(face)
        
        return np.array(processed)
    
    def create_morphing_dataset(self, num_pairs: int = 10, 
                               target_size: Tuple[int, int] = (256, 256)) -> Dict[str, Any]:
        """Crée un dataset prêt pour l'entraînement de morphing"""
        print(f"🔄 Création d'un dataset de morphing avec {num_pairs} paires...")
        
        # Collecter les paires
        pairs = self.get_optimal_face_pairs(num_pairs)
        
        # Séparer les visages
        faces1 = [pair[0] for pair in pairs]
        faces2 = [pair[1] for pair in pairs]
        infos = [pair[2] for pair in pairs]
        
        # Préprocesser
        faces1_batch = self.preprocess_batch(faces1, target_size)
        faces2_batch = self.preprocess_batch(faces2, target_size)
        
        dataset = {
            'source_faces': faces1_batch,
            'target_faces': faces2_batch,
            'pairs_info': infos,
            'num_pairs': num_pairs,
            'face_shape': target_size + (3,),  # RGB
            'ready_for_training': True
        }
        
        print(f"✅ Dataset créé: {num_pairs} paires de {target_size}")
        return dataset
    
    def visualize_morphing_candidates(self, num_pairs: int = 3) -> None:
        """Visualise les candidats pour le morphing - VERSION CORRIGÉE"""
        try:
            pairs = self.get_optimal_face_pairs(num_pairs)
            
            fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 4*num_pairs))
            if num_pairs == 1:
                axes = axes.reshape(1, -1)
            
            for i, (face1, face2, info) in enumerate(pairs):
                # Préparer les images pour l'affichage
                display_face1 = self._prepare_for_display(face1)
                display_face2 = self._prepare_for_display(face2)
                
                # Afficher face1
                axes[i, 0].imshow(display_face1, cmap='gray' if len(display_face1.shape) == 2 else None)
                axes[i, 0].set_title(f"Personne {info.get('person1_id', '?')}")
                axes[i, 0].axis('off')
                
                # Afficher face2
                axes[i, 1].imshow(display_face2, cmap='gray' if len(display_face2.shape) == 2 else None)
                axes[i, 1].set_title(f"Personne {info.get('person2_id', '?')}")
                axes[i, 1].axis('off')
            
            plt.suptitle("Candidats pour le morphing de visages", fontsize=14)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Erreur lors de la visualisation: {e}")
            print("Essai avec des images individuelles...")
            self._fallback_visualization(num_pairs)
    
    def _prepare_for_display(self, face: np.ndarray) -> np.ndarray:
        """Prépare une image pour l'affichage matplotlib"""
        # Convertir en float32
        display_face = face.astype(np.float32)
        
        # Normaliser si nécessaire
        if display_face.max() > 1.0:
            display_face = display_face / 255.0
            
        # Clipper les valeurs
        display_face = np.clip(display_face, 0.0, 1.0)
        
        return display_face
    
    def _fallback_visualization(self, num_pairs: int) -> None:
        """Visualisation de secours en cas d'erreur"""
        try:
            # Charger quelques images directement
            if self.loader.faces is None:
                self.loader.load_data()
                
            # Prendre les premières images de différentes personnes
            fig, axes = plt.subplots(1, min(num_pairs * 2, 6), figsize=(12, 4))
            if len(axes.shape) == 0:
                axes = [axes]
            
            for i in range(min(num_pairs * 2, 6)):
                if i < len(self.loader.faces):
                    face = self.loader.faces[i]
                    display_face = self._prepare_for_display(face)
                    
                    ax = axes[i] if len(axes) > 1 else axes
                    ax.imshow(display_face, cmap='gray')
                    ax.set_title(f"Personne {self.loader.labels[i]}")
                    ax.axis('off')
            
            plt.suptitle("Échantillons du dataset", fontsize=14)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Erreur de visualisation de secours: {e}")
    
    def get_person_variations(self, person_id: int, max_count: int = 5) -> List[np.ndarray]:
        """Récupère les variations d'une personne (images augmentées)"""
        all_faces = self.loader.get_face_by_person(person_id)
        
        # Limiter le nombre et préprocesser
        selected = all_faces[:max_count] if len(all_faces) > max_count else all_faces
        processed = []
        
        for face in selected:
            proc_face = self.loader.preprocess_for_morphing(face)
            processed.append(proc_face)
        
        return processed
    
    def create_augmentation_showcase(self, person_id: int = 0) -> None:
        """Montre les différentes augmentations d'une personne"""
        variations = self.get_person_variations(person_id, 8)
        
        if len(variations) == 0:
            print(f"❌ Aucune image trouvée pour la personne {person_id}")
            return
        
        rows = int(np.ceil(len(variations) / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(12, 3*rows))
        axes = axes.flatten()
        
        for i, face in enumerate(variations):
            if i < len(axes):
                # Normaliser pour l'affichage
                display_face = face.astype(np.float32)
                if display_face.max() > 1.0:
                    display_face = np.clip(display_face / 255.0, 0.0, 1.0)
                
                if len(display_face.shape) == 3:
                    axes[i].imshow(display_face)
                else:
                    axes[i].imshow(display_face, cmap='gray')
                axes[i].set_title(f"Variation {i+1}")
                axes[i].axis('off')
        
        # Cacher axes inutilisés
        for i in range(len(variations), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Variations de la personne {person_id} (augmentations)", fontsize=14)
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation rapide - VERSION SÉCURISÉE
def demo_morphing_integration():
    """Démonstration de l'intégration pour le morphing - VERSION CORRIGÉE"""
    metadata_path = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    
    print("🎭 Démonstration de l'intégration pour le morphing")
    print("=" * 50)
    
    try:
        # Créer l'assistant morphing
        helper = MorphingDatasetHelper(metadata_path)
        
        # 1. Visualiser les candidats avec gestion d'erreur
        print("👀 Visualisation des candidats...")
        helper.visualize_morphing_candidates(3)
        
        # 2. Montrer les variations d'augmentation
        print("🔄 Showcase des augmentations...")
        helper.create_augmentation_showcase(0)
        
        # 3. Créer un dataset de morphing
        print("📦 Création du dataset de morphing...")
        morphing_dataset = helper.create_morphing_dataset(5)
        
        print("\n✅ Intégration morphing terminée!")
        print(f"Dataset prêt avec {morphing_dataset['num_pairs']} paires")
        print(f"Forme des visages: {morphing_dataset['face_shape']}")
        
        return helper, morphing_dataset
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        print("Exécution du mode de test simplifié...")
        return simple_test_mode()

def simple_test_mode():
    """Mode de test simplifié en cas d'erreur"""
    metadata_path = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    loader = OlivettiDatasetLoader(metadata_path)
    
    # Charger juste les données
    faces, labels = loader.load_data()
    
    # Test simple
    print(f"✅ Test simplifié réussi: {faces.shape} images chargées")
    return None, {"status": "simple_test_completed"}

if __name__ == "__main__":
    helper, dataset = demo_morphing_integration()
