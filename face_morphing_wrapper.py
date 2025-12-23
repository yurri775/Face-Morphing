import numpy as np
from dataset_loader import OlivettiDatasetLoader
from morphing_integration import MorphingDatasetHelper
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class FaceMorphingWrapper:
    """Wrapper pour intégrer le dataset Olivetti avec vos algorithmes de morphing"""
    
    def __init__(self, metadata_file: Optional[str] = None):
        self.metadata_file = metadata_file
        self.helper = MorphingDatasetHelper(metadata_file)
        self.current_pair = None
        self.morphing_cache = {}
    
    def get_face_pair_for_morphing(self, method: str = "different_persons") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Récupère une paire de visages prête pour le morphing"""
        face1, face2, info = self.helper.loader.get_morphing_pair(method)
        self.current_pair = (face1, face2, info)
        
        print(f"✅ Paire préparée: Personne {info.get('person1_id', '?')} ↔ Personne {info.get('person2_id', '?')}")
        print(f"Forme des images: {face1.shape}")
        
        return face1, face2, info
    
    def preprocess_for_your_morphing_algorithm(self, face1: np.ndarray, face2: np.ndarray, 
                                             target_size: Tuple[int, int] = (256, 256),
                                             grayscale: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Préprocess les images selon les besoins de votre algorithme de morphing
        
        Args:
            face1, face2: Images d'entrée
            target_size: Taille cible (largeur, hauteur)
            grayscale: Si True, convertit en niveaux de gris
        """
        # Assurer le bon format
        face1 = face1.astype(np.float32)
        face2 = face2.astype(np.float32)
        
        # Redimensionner si nécessaire
        if face1.shape[:2] != target_size:
            face1 = cv2.resize(face1, target_size)
        if face2.shape[:2] != target_size:
            face2 = cv2.resize(face2, target_size)
        
        # Convertir en niveaux de gris si demandé
        if grayscale and len(face1.shape) == 3:
            face1 = np.mean(face1, axis=2)
            face2 = np.mean(face2, axis=2)
        
        return face1, face2
    
    def create_morphing_sequence_data(self, num_pairs: int = 10) -> Dict[str, Any]:
        """Crée un ensemble de données pour une séquence de morphing"""
        morphing_data = {
            'pairs': [],
            'preprocessed_pairs': [],
            'metadata': []
        }
        
        for i in range(num_pairs):
            # Obtenir une nouvelle paire
            face1, face2, info = self.get_face_pair_for_morphing()
            
            # Préprocesser pour votre algorithme
            proc_face1, proc_face2 = self.preprocess_for_your_morphing_algorithm(face1, face2)
            
            morphing_data['pairs'].append((face1, face2))
            morphing_data['preprocessed_pairs'].append((proc_face1, proc_face2))
            morphing_data['metadata'].append(info)
        
        print(f"✅ Données de morphing créées: {num_pairs} paires")
        return morphing_data
    
    def visualize_morphing_preparation(self, face1: np.ndarray, face2: np.ndarray, 
                                     info: Dict, steps: int = 5) -> None:
        """Visualise la préparation du morphing avec étapes intermédiaires simulées"""
        fig, axes = plt.subplots(2, steps + 2, figsize=(15, 6))
        
        # Ligne 1: Images originales et étapes
        axes[0, 0].imshow(face1, cmap='gray' if len(face1.shape) == 2 else None)
        axes[0, 0].set_title(f"Source\nPersonne {info.get('person1_id', '?')}")
        axes[0, 0].axis('off')
        
        # Étapes intermédiaires simulées
        for i in range(1, steps + 1):
            alpha = i / (steps + 1)
            morphed = (1 - alpha) * face1 + alpha * face2
            morphed = np.clip(morphed, 0, 1)
            
            axes[0, i].imshow(morphed, cmap='gray' if len(morphed.shape) == 2 else None)
            axes[0, i].set_title(f"Étape {i}\n({alpha:.1%})")
            axes[0, i].axis('off')
        
        axes[0, -1].imshow(face2, cmap='gray' if len(face2.shape) == 2 else None)
        axes[0, -1].set_title(f"Cible\nPersonne {info.get('person2_id', '?')}")
        axes[0, -1].axis('off')
        
        # Ligne 2: Informations et statistiques
        for j, ax in enumerate(axes[1]):
            ax.axis('off')
            if j == 0:
                ax.text(0.5, 0.5, f"Forme: {face1.shape}\nType: {face1.dtype}\nMin/Max: {face1.min():.3f}/{face1.max():.3f}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
            elif j == len(axes[1]) - 1:
                ax.text(0.5, 0.5, f"Forme: {face2.shape}\nType: {face2.dtype}\nMin/Max: {face2.min():.3f}/{face2.max():.3f}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        plt.suptitle("Préparation du Morphing avec Dataset Olivetti", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def export_for_external_morphing_tool(self, output_dir: str = "morphing_export") -> str:
        """Exporte les données dans un format standard pour outils externes"""
        export_path = Path(output_dir)
        export_path.mkdir(exist_ok=True)
        
        # Obtenir quelques paires
        morphing_data = self.create_morphing_sequence_data(5)
        
        # Exporter chaque paire
        for i, ((face1, face2), info) in enumerate(zip(morphing_data['pairs'], morphing_data['metadata'])):
            # Sauvegarder les images
            face1_path = export_path / f"pair_{i}_source_person_{info.get('person1_id', 'unknown')}.png"
            face2_path = export_path / f"pair_{i}_target_person_{info.get('person2_id', 'unknown')}.png"
            
            # Convertir pour la sauvegarde
            face1_save = (face1 * 255).astype(np.uint8)
            face2_save = (face2 * 255).astype(np.uint8)
            
            cv2.imwrite(str(face1_path), face1_save)
            cv2.imwrite(str(face2_path), face2_save)
        
        # Créer un fichier de métadonnées
        metadata_file = export_path / "morphing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'dataset_info': 'Olivetti Faces Augmented Dataset',
                'total_pairs': len(morphing_data['pairs']),
                'pairs_metadata': morphing_data['metadata'],
                'export_format': 'PNG images, normalized to [0,255] uint8'
            }, f, indent=2)
        
        print(f"📁 Export terminé dans: {export_path}")
        print(f"📄 Métadonnées: {metadata_file}")
        
        return str(export_path)

# Test rapide du wrapper
def test_morphing_wrapper():
    """Test rapide du wrapper de morphing"""
    print("🧪 Test du wrapper de morphing")
    print("=" * 40)
    
    metadata_path = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    
    try:
        # Créer le wrapper
        wrapper = FaceMorphingWrapper(metadata_path)
        
        # Obtenir une paire
        face1, face2, info = wrapper.get_face_pair_for_morphing()
        
        # Visualiser la préparation
        wrapper.visualize_morphing_preparation(face1, face2, info)
        
        # Exporter pour outils externes
        export_path = wrapper.export_for_external_morphing_tool()
        
        print("✅ Test du wrapper terminé avec succès!")
        return wrapper
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        return None

if __name__ == "__main__":
    wrapper = test_morphing_wrapper()
