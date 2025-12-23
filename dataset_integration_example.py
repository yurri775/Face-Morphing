from enhanced_dataset_loader import EnhancedOlivettiLoader
from metadata_parser import DatasetMetadata
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Démonstration complète d'utilisation avec métadonnées"""
    
    # Chemin vers le fichier de métadonnées
    metadata_file = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    
    print("=== Initialisation du Dataset avec Métadonnées ===")
    
    # 1. Créer le chargeur avec métadonnées
    loader = EnhancedOlivettiLoader(metadata_file=metadata_file)
    
    # 2. Télécharger et valider
    path, validation = loader.download_and_validate()
    
    if not validation.get('all_valid', False):
        print("⚠️ Attention: Problèmes de validation détectés")
        return
    
    # 3. Afficher les statistiques
    loader.print_statistics()
    
    # 4. Créer le rapport
    report_path = "dataset_report.md"
    report = loader.create_metadata_report(report_path)
    
    # 5. Exemples d'utilisation avancée
    print("\n=== Exemples d'Utilisation ===")
    
    # Récupérer images d'une personne (originales uniquement)
    person_0_original = loader.get_face_by_person(0, include_augmented=False)
    person_0_all = loader.get_face_by_person(0, include_augmented=True)
    
    print(f"Personne 0 - Images originales: {len(person_0_original)}")
    print(f"Personne 0 - Toutes images: {len(person_0_all)}")
    
    # Afficher comparaison
    if len(person_0_all) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        for i, ax in enumerate(axes.flat):
            if i < len(person_0_all):
                face = person_0_all[i]
                if face.max() <= 1.0:
                    ax.imshow(face, cmap='gray')
                else:
                    ax.imshow(face / 255.0, cmap='gray')
                    
                title = "Originale" if i < len(person_0_original) else "Augmentée"
                ax.set_title(f"Image {i+1} - {title}")
                ax.axis('off')
        
        plt.suptitle("Comparaison Images Originales vs Augmentées")
        plt.tight_layout()
        plt.show()
    
    print("\n✅ Intégration du dataset terminée avec succès!")
    print(f"📄 Consultez le rapport détaillé: {report_path}")

if __name__ == "__main__":
    main()
