from dataset_loader import OlivettiDatasetLoader
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Exemple rapide d'utilisation du dataset Olivetti"""
    
    # Chemin vers les métadonnées (ajustez selon votre installation)
    metadata_path = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    
    print("🚀 Démarrage rapide avec le dataset Olivetti")
    print("=" * 50)
    
    # 1. Créer le chargeur avec métadonnées
    loader = OlivettiDatasetLoader(metadata_file=metadata_path)
    
    # 2. Le dataset sera téléchargé automatiquement au premier accès
    print("\n📥 Chargement du dataset...")
    faces, labels = loader.load_data()
    
    # 3. Afficher une galerie d'exemples
    print("\n🖼️  Affichage de la galerie...")
    loader.display_sample_gallery(count=8)
    
    # 4. Préparer une paire pour le morphing
    print("\n🔀 Préparation d'une paire pour le morphing...")
    face1, face2, info = loader.get_morphing_pair("different_persons")
    
    print(f"✓ Paire prête: Personne {info['person1_id']} ↔ Personne {info['person2_id']}")
    print(f"✓ Forme des visages: {face1.shape}")
    
    # 5. Afficher la paire sélectionnée
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Normaliser les images pour l'affichage
    display_face1 = face1.astype(np.float32)
    display_face2 = face2.astype(np.float32)
    
    if display_face1.max() > 1.0:
        display_face1 = np.clip(display_face1 / 255.0, 0.0, 1.0)
    if display_face2.max() > 1.0:
        display_face2 = np.clip(display_face2 / 255.0, 0.0, 1.0)
    
    # Afficher selon le nombre de canaux
    if len(display_face1.shape) == 3:
        axes[0].imshow(display_face1)
    else:
        axes[0].imshow(display_face1, cmap='gray')
    axes[0].set_title(f"Personne {info['person1_id']}")
    axes[0].axis('off')
    
    if len(display_face2.shape) == 3:
        axes[1].imshow(display_face2)
    else:
        axes[1].imshow(display_face2, cmap='gray')
    axes[1].set_title(f"Personne {info['person2_id']}")
    axes[1].axis('off')
    
    plt.suptitle("Paire sélectionnée pour le morphing", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 6. Exporter les informations
    loader.export_dataset_info()
    
    print("\n✅ Configuration terminée!")
    print("Vous pouvez maintenant utiliser face1 et face2 pour votre algorithme de morphing.")
    print("📄 Consultez dataset_info.txt pour plus de détails.")

    return loader, face1, face2, info

if __name__ == "__main__":
    loader, face1, face2, info = main()
