import kagglehub
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from metadata_parser import DatasetMetadata

class EnhancedOlivettiLoader:
    """Chargeur de dataset Olivetti avec gestion des métadonnées"""
    
    def __init__(self, metadata_file: Optional[str] = None, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_path = None
        self.faces = None
        self.labels = None
        
        # Charger les métadonnées si disponibles
        self.metadata = DatasetMetadata(metadata_file) if metadata_file else None
        
    def download_and_validate(self) -> Tuple[str, Dict[str, bool]]:
        """Télécharge le dataset et valide avec les métadonnées"""
        print("=== Téléchargement du Dataset ===")
        path = kagglehub.dataset_download("martininf1n1ty/olivetti-faces-augmented-dataset")
        self.dataset_path = Path(path)
        
        if self.metadata:
            print(self.metadata.get_dataset_summary())
            
        # Charger et valider
        faces, labels = self.load_data()
        validation = {}
        
        if self.metadata:
            print("\n=== Validation des Données ===")
            validation = self.metadata.validate_dataset(faces, labels)
            
            for check, result in validation.items():
                status = "✓" if result else "✗"
                print(f"{status} {check}: {result}")
                
            if validation.get('all_valid', False):
                print("\n✅ Toutes les validations sont passées!")
            else:
                print("\n⚠️  Certaines validations ont échoué.")
        
        return path, validation
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Charge les données avec logging amélioré"""
        if self.dataset_path is None:
            self.download_and_validate()
            
        faces_path = self.dataset_path / "augmented_faces.npy"
        labels_path = self.dataset_path / "augmented_labels.npy"
        
        if not faces_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Fichiers du dataset introuvables")
            
        print("Chargement des données...")
        self.faces = np.load(faces_path)
        self.labels = np.load(labels_path)
        
        print(f"✓ Images chargées: {self.faces.shape}")
        print(f"✓ Labels chargés: {self.labels.shape}")
        print(f"✓ Type de données: {self.faces.dtype}")
        print(f"✓ Plage de valeurs: [{self.faces.min():.3f}, {self.faces.max():.3f}]")
        
        return self.faces, self.labels
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques détaillées du dataset"""
        if self.faces is None or self.labels is None:
            self.load_data()
            
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        stats = {
            'total_images': len(self.faces),
            'image_shape': self.faces.shape[1:],
            'data_type': str(self.faces.dtype),
            'value_range': (float(self.faces.min()), float(self.faces.max())),
            'total_persons': len(unique_labels),
            'images_per_person': {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts))
            },
            'memory_usage_mb': float(self.faces.nbytes / 1024 / 1024),
            'augmentation_info': self.metadata.get_augmentation_info() if self.metadata else None
        }
        
        return stats
    
    def print_statistics(self) -> None:
        """Affiche les statistiques du dataset"""
        stats = self.get_dataset_statistics()
        
        print("\n=== Statistiques du Dataset ===")
        print(f"📊 Total d'images: {stats['total_images']}")
        print(f"📐 Forme des images: {stats['image_shape']}")
        print(f"🔢 Type de données: {stats['data_type']}")
        print(f"📏 Plage de valeurs: [{stats['value_range'][0]:.3f}, {stats['value_range'][1]:.3f}]")
        print(f"👥 Nombre de personnes: {stats['total_persons']}")
        print(f"📷 Images par personne (moyenne): {stats['images_per_person']['mean']:.1f}")
        print(f"💾 Utilisation mémoire: {stats['memory_usage_mb']:.1f} MB")
        
        if stats['augmentation_info']:
            print(f"\n{stats['augmentation_info']}")
    
    def get_face_by_person(self, person_id: int, include_augmented: bool = True) -> List[np.ndarray]:
        """Récupère les images d'une personne avec options d'augmentation"""
        if self.faces is None or self.labels is None:
            self.load_data()
            
        mask = self.labels == person_id
        person_faces = self.faces[mask]
        
        if not include_augmented and self.metadata:
            # Si on ne veut que les originales, prendre seulement les 10 premières
            # (approximation, les images originales sont généralement en premier)
            person_faces = person_faces[:10] if len(person_faces) >= 10 else person_faces
            
        return person_faces
    
    def create_metadata_report(self, output_file: Optional[str] = None) -> str:
        """Crée un rapport détaillé sur le dataset"""
        if not self.metadata:
            return "Métadonnées non disponibles"
            
        report = f"""
# Rapport du Dataset Olivetti Faces Augmenté

{self.metadata.get_dataset_summary()}

## Statistiques Calculées
"""
        
        if self.faces is not None:
            stats = self.get_dataset_statistics()
            report += f"""
- **Images chargées**: {stats['total_images']}
- **Forme des images**: {stats['image_shape']}
- **Utilisation mémoire**: {stats['memory_usage_mb']:.1f} MB
- **Personnes détectées**: {stats['total_persons']}
"""
        
        report += f"\n{self.metadata.get_augmentation_info()}"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Rapport sauvegardé dans: {output_file}")
            
        return report
