from dataset_loader import OlivettiDatasetLoader
import numpy as np

def debug_dataset_types():
    """Script de débogage pour analyser les types de données"""
    
    metadata_path = r"c:\Users\marwa\Downloads\olivetti-faces-augmented-dataset-metadata.json"
    loader = OlivettiDatasetLoader(metadata_file=metadata_path)
    
    print("🔍 Analyse des types de données du dataset")
    print("=" * 50)
    
    # Charger les données
    faces, labels = loader.load_data()
    
    # Analyse détaillée
    print(f"📊 Forme des images: {faces.shape}")
    print(f"📊 Type de données: {faces.dtype}")
    print(f"📊 Plage de valeurs: [{faces.min():.6f}, {faces.max():.6f}]")
    print(f"📊 Moyenne: {faces.mean():.6f}")
    print(f"📊 Écart-type: {faces.std():.6f}")
    
    # Test d'une image
    print(f"\n🔬 Analyse d'une image individuelle:")
    test_face = faces[0]
    print(f"Forme: {test_face.shape}")
    print(f"Type: {test_face.dtype}")
    print(f"Valeurs min/max: [{test_face.min():.6f}, {test_face.max():.6f}]")
    
    # Test de conversion
    print(f"\n🛠️  Tests de conversion:")
    try:
        # Test conversion float32
        face_f32 = test_face.astype(np.float32)
        print(f"✅ Conversion float32: {face_f32.dtype}")
        
        # Test normalisation
        if face_f32.max() > 1.0:
            face_norm = face_f32 / 255.0
        else:
            face_norm = face_f32
        face_norm = np.clip(face_norm, 0.0, 1.0)
        print(f"✅ Normalisation: [{face_norm.min():.3f}, {face_norm.max():.3f}]")
        
        # Test conversion couleur
        if len(face_norm.shape) == 2:
            face_rgb = np.stack([face_norm, face_norm, face_norm], axis=-1)
            print(f"✅ Conversion RGB: {face_rgb.shape}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion: {e}")
    
    print(f"\n✅ Analyse terminée!")

if __name__ == "__main__":
    debug_dataset_types()
