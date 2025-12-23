"""
Extrait toutes les vidéos MP4 de results/morph_all vers des dossiers d'images PNG.
Par défaut, chaque vidéo "morph_X_Y.mp4" produit des images dans
"results/morph_all/frames/morph_X_Y/frame_00001.png".

Usage:
    python results/extract_frames.py [--fps N] [--overwrite]

Options:
    --fps N        (optionnel) impose un fps de sortie (sinon fps natif)
    --overwrite    écrase les images existantes
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MORPH_DIR = ROOT / "morph_all"
FRAMES_DIR = MORPH_DIR / "frames"


def extract_video(video_path: Path, fps: float | None, overwrite: bool) -> bool:
    if not video_path.exists():
        print(f"⚠️  Fichier manquant: {video_path}")
        return False

    stem = video_path.stem  # e.g. morph_02_03
    out_dir = FRAMES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "frame_%05d.png"

    if not overwrite and any(out_dir.iterdir()):
        print(f"⏩ Skip (déjà extrait): {out_dir}")
        return True

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
    ]
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [str(out_pattern)]

    print(f"▶️  Extraction: {video_path.name} -> {out_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Échec ffmpeg pour {video_path.name}\n{result.stderr}")
        return False

    print(f"✅ OK: {out_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=None, help="FPS de sortie (optionnel)")
    parser.add_argument("--overwrite", action="store_true", help="Écraser les images existantes")
    args = parser.parse_args()

    if not MORPH_DIR.exists():
        raise SystemExit(f"Dossier introuvable: {MORPH_DIR}")

    videos = sorted(p for p in MORPH_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"Aucune vidéo .mp4 trouvée dans {MORPH_DIR}")

    ok = 0
    fail = 0
    for vid in videos:
        if extract_video(vid, args.fps, args.overwrite):
            ok += 1
        else:
            fail += 1

    print(f"Terminé. Succès: {ok}, Échecs: {fail}")


if __name__ == "__main__":
    main()
