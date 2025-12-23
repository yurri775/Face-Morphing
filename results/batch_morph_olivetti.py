"""
Génère des vidéos de morphing pour des paires d'images du dataset Olivetti exporté.
- Utilise le pipeline existant code/__init__.py
- Sélectionne la première image de chaque personne (index 0, 50, 100, ...)
- Morph entre personne i et i+1 (0->1, 1->2, ... 38->39)
- Sauvegarde dans results/morph_all/morph_{pid1:02d}_{pid2:02d}.mp4

Usage :
    python results/batch_morph_olivetti.py
Options :
    --start <int>  premier id personne (par défaut 0)
    --end <int>    dernier id personne (par défaut 39)
    --skip-existing  saute les sorties déjà présentes

Note : cela peut prendre du temps (40 vidéos). Assurez-vous d'avoir ffmpeg et le modèle dlib téléchargé.
"""

import argparse
import subprocess
from pathlib import Path

def first_image_for_person(pid: int) -> Path:
    base = pid * 50  # car 50 images par personne dans l'export
    return Path(f"images/person_{pid:02d}_img_{base:04d}.png")


def run_morph(img1: Path, img2: Path, out_path: Path) -> bool:
    if not img1.exists() or not img2.exists():
        print(f"⚠️  Fichier manquant: {img1 if not img1.exists() else img2}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "code/__init__.py",
        "--img1", str(img1),
        "--img2", str(img2),
        "--output", str(out_path),
    ]
    print(f"▶️  Morphing {img1.name} -> {img2.name} -> {out_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Échec ({img1.name},{img2.name})\n{result.stderr}")
        return False
    print("✅ OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=39)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_root = Path("results/morph_all")
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for pid in range(args.start, args.end):
        img1 = first_image_for_person(pid)
        img2 = first_image_for_person(pid + 1)
        out_path = out_root / f"morph_{pid:02d}_{pid+1:02d}.mp4"
        jobs.append((img1, img2, out_path))

    ok, fail = 0, 0
    for img1, img2, out_path in jobs:
        if args.skip_existing and out_path.exists():
            print(f"⏩ Skip (existe): {out_path}")
            continue
        if run_morph(img1, img2, out_path):
            ok += 1
        else:
            fail += 1

    print(f"Terminé. Succès: {ok}, Échecs: {fail}")
    if fail:
        print("Consultez les logs ci-dessus pour les paires en échec.")


if __name__ == "__main__":
    main()
