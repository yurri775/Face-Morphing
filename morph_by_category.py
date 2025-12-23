"""
Morphe les images AU SEIN de chaque catégorie (pas de mélange entre catégories).
Résultats sous results/morph_results/<categorie>/
"""
import os
import sys
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_ROOT = PROJECT_ROOT / "images" / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "morph_results"

# Ajoute le dossier code au path Python
CODE_DIR = PROJECT_ROOT / "code"
if CODE_DIR.exists():
    sys.path.append(str(CODE_DIR))

# Détection dynamique de l'API de morphing
_morph_api = {"mode": "none", "callable": None}

def _init_morph_api():
    global _morph_api
    try:
        import face_morph as fm
        if hasattr(fm, "FaceMorpher"):
            try:
                instance = fm.FaceMorpher()
                def _call(img1, img2, alpha):
                    return instance.morph(img1, img2, alpha)
                _morph_api = {"mode": "class", "callable": _call}
                return
            except Exception:
                pass
        if hasattr(fm, "morph"):
            def _call(img1, img2, alpha):
                return fm.morph(img1, img2, alpha)
            _morph_api = {"mode": "func", "callable": _call}
            return
        if hasattr(fm, "morph_images"):
            def _call(img1, img2, alpha):
                return fm.morph_images(img1, img2, alpha)
            _morph_api = {"mode": "func2", "callable": _call}
            return
    except Exception:
        pass
    _morph_api = {"mode": "fallback", "callable": None}

def cross_dissolve(img1, img2, alpha):
    """Fondu croisé simple si l'API de morph n'est pas disponible"""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1r = cv2.resize(img1, (w, h))
    img2r = cv2.resize(img2, (w, h))
    return cv2.addWeighted(img1r, 1.0 - alpha, img2r, alpha, 0)

def safe_imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {path}")
    return img

def list_images(cat_dir: Path):
    """Liste toutes les images d'un dossier catégorie"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(str(cat_dir / ext)))
    return sorted(files)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def generate_gif_from_frames(frames_dir: Path, gif_path: Path, fps: int = 12):
    """Génère un GIF à partir des frames"""
    try:
        import imageio
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        if not frames:
            return False
        images = [imageio.v2.imread(str(f)) for f in frames]
        duration = 1.0 / max(fps, 1)
        imageio.mimsave(str(gif_path), images, duration=duration)
        return True
    except Exception:
        return False

def morph_sequence(img1_path: Path, img2_path: Path, out_dir: Path, num_frames: int, make_gif: bool, pair_id: str = ""):
    """Crée une séquence de morph entre deux images"""
    ensure_dir(out_dir)
    img1 = safe_imread(img1_path)
    img2 = safe_imread(img2_path)

    _init_morph_api()
    use_api = _morph_api["mode"] != "fallback"
    call_api = _morph_api["callable"]

    for i in range(num_frames + 1):
        alpha = i / float(num_frames)
        try:
            if use_api and call_api:
                morphed = call_api(img1, img2, alpha)
                if morphed is None:
                    morphed = cross_dissolve(img1, img2, alpha)
            else:
                morphed = cross_dissolve(img1, img2, alpha)
        except Exception:
            morphed = cross_dissolve(img1, img2, alpha)

        frame_path = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), morphed)

    if make_gif:
        gif_ok = generate_gif_from_frames(out_dir, out_dir / "morph.gif")
        return gif_ok
    return True

def morph_category_images(category: str, images_root: Path, results_root: Path, 
                         frames: int, make_gif: bool, sequential: bool = True):
    """
    Morphe les images au sein d'une même catégorie.
    
    Si sequential=True: image[0]->image[1], image[1]->image[2], etc.
    Si sequential=False: toutes les paires (image[0]->image[1], image[0]->image[2], etc.)
    """
    cat_dir = images_root / category
    images = list_images(cat_dir)
    
    if not images or len(images) < 2:
        print(f"[SKIP] '{category}': moins de 2 images trouvées")
        return 0
    
    cat_results = results_root / category.replace(" ", "_")
    ensure_dir(cat_results)
    
    total_morphs = 0
    
    if sequential:
        # Mode séquentiel: image[i] -> image[i+1]
        for i in range(len(images) - 1):
            pair_dir = cat_results / f"morph_{i:02d}_{i+1:02d}"
            img1_name = Path(images[i]).stem
            img2_name = Path(images[i+1]).stem
            try:
                ok = morph_sequence(Path(images[i]), Path(images[i+1]), 
                                   pair_dir, frames, make_gif, f"{i}-{i+1}")
                if ok:
                    total_morphs += 1
                    print(f"  [OK] {img1_name} -> {img2_name}")
                else:
                    print(f"  [WARN] GIF non généré: {pair_dir}")
            except Exception as e:
                print(f"  [ERR] {img1_name} -> {img2_name}: {e}")
    else:
        # Mode toutes les paires: image[i] -> image[j] pour tous i < j
        pair_id = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                pair_dir = cat_results / f"pair_{pair_id:03d}"
                img1_name = Path(images[i]).stem
                img2_name = Path(images[j]).stem
                try:
                    ok = morph_sequence(Path(images[i]), Path(images[j]), 
                                       pair_dir, frames, make_gif, f"{i}-{j}")
                    if ok:
                        total_morphs += 1
                        print(f"  [OK] {img1_name} -> {img2_name}")
                    else:
                        print(f"  [WARN] GIF non généré: {pair_dir}")
                except Exception as e:
                    print(f"  [ERR] {img1_name} -> {img2_name}: {e}")
                pair_id += 1
    
    return total_morphs

def get_all_categories(images_root: Path):
    """Liste toutes les catégories disponibles"""
    if not images_root.exists():
        return []
    subdirs = [d.name for d in images_root.iterdir() if d.is_dir()]
    return sorted(subdirs)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Morphe les images au sein de chaque catégorie (sans mélange)"
    )
    ap.add_argument("--mode", choices=["sequential", "all-pairs"], default="sequential",
                    help="sequential: img[i]->img[i+1], all-pairs: toutes les paires")
    ap.add_argument("--frames", type=int, default=10,
                    help="Nombre de frames par morph")
    ap.add_argument("--gif", action="store_true",
                    help="Générer un GIF dans chaque dossier de morph")
    ap.add_argument("--categories", type=str, default="",
                    help="Catégories à traiter (ex: 'white men,white women'); vide = toutes")
    return ap.parse_args()

def main():
    args = parse_args()

    if not IMAGES_ROOT.exists():
        print(f"Erreur: Dossier images introuvable: {IMAGES_ROOT}")
        sys.exit(1)

    ensure_dir(RESULTS_ROOT)

    # Récupère les catégories à traiter
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    else:
        categories = get_all_categories(IMAGES_ROOT)

    if not categories:
        print("Erreur: Aucune catégorie trouvée!")
        sys.exit(1)

    sequential = (args.mode == "sequential")

    print("=" * 70)
    print(f"Morphing par catégorie (mode: {args.mode})")
    print(f"Frames par morph: {args.frames}")
    print(f"GIF: {'OUI' if args.gif else 'NON'}")
    print("=" * 70)

    grand_total = 0
    for category in categories:
        cat_dir = IMAGES_ROOT / category
        if not cat_dir.exists():
            print(f"\n[SKIP] Catégorie introuvable: {category}")
            continue
        
        images = list_images(cat_dir)
        print(f"\n📁 {category}: {len(images)} image(s)")
        
        total = morph_category_images(category, IMAGES_ROOT, RESULTS_ROOT, 
                                     args.frames, args.gif, sequential)
        grand_total += total
        print(f"   → {total} morph(s) créé(s)")

    print("\n" + "=" * 70)
    print(f"✅ Terminé! {grand_total} morph(s) au total dans {RESULTS_ROOT}")
    print("=" * 70)

if __name__ == "__main__":
    main()