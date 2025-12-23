import cv2
from pathlib import Path

pairs = [
    (Path('images/person_20_img_1000.png'), Path('results/tmp_img1.png')),
    (Path('images/person_30_img_1500.png'), Path('results/tmp_img2.png')),
]

for src, dst in pairs:
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f'Missing image: {src}')
    up = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), up)
    print(f'Upscaled {src} -> {dst}')
