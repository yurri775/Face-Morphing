"""
Exporte toutes les images du dataset Olivetti Faces augmenté vers le dossier images/.
Usage :
    python export_olivetti_images.py

Options :
    --metadata <chemin>  (optionnel) chemin du fichier de métadonnées JSON
    --out <dossier>      (optionnel) dossier de sortie (par défaut: images)
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dataset_loader import OlivettiDatasetLoader


def to_uint8_rgb(face: np.ndarray) -> np.ndarray:
    """Convertit une image du dataset en uint8 RGB prête à sauvegarder."""
    arr = face.astype(np.float32)

    # Normaliser dans [0, 1] puis mettre en [0, 255]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    # Assurer 3 canaux pour la sauvegarde couleur (OpenCV accepte aussi 1 canal)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr


def export_images(metadata_file: Optional[str], output_dir: str) -> None:
    loader = OlivettiDatasetLoader(metadata_file=metadata_file)
    faces, labels = loader.load_data()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(faces)
    print(f"➡️ Export des {total} images vers {out_path.resolve()}")

    for idx, (face, label) in enumerate(zip(faces, labels)):
        img = to_uint8_rgb(face)
        filename = out_path / f"person_{int(label):02d}_img_{idx:04d}.png"
        cv2.imwrite(str(filename), img)

        if (idx + 1) % 200 == 0 or idx + 1 == total:
            print(f"  - {idx + 1}/{total} images exportées")

    print("✅ Export terminé")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export des images Olivetti augmentées")
    parser.add_argument("--metadata", type=str, default=None, help="Chemin du fichier de métadonnées JSON (optionnel)")
    parser.add_argument("--out", type=str, default="images", help="Dossier de sortie (par défaut: images)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_images(args.metadata, args.out)


if __name__ == "__main__":
    main()
