from pathlib import Path
import re
import cv2

def numeric_key(p: Path):
    nums = re.findall(r'\d+', p.stem)
    return (int(nums[-1]) if nums else 10**12, p.name.lower())

def prepare_case(raw_dir: str, out_root: str):
    raw = Path(raw_dir)
    out = Path(out_root)
    frames = out / "frames"
    frames_0p5 = out / "frames_0p5"
    frames.mkdir(parents=True, exist_ok=True)
    frames_0p5.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = sorted([p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in exts], key=numeric_key)

    n = 0
    for i, p in enumerate(imgs, start=1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        name = f"{i:06d}.jpg"  # standard naming
        cv2.imwrite(str(frames / name), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        h, w = im.shape[:2]
        im_half = cv2.resize(im, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_0p5 / name), im_half, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1

    print(f"{out_root}: wrote {n} frames")

prepare_case(
    "/home/aaron/Downloads/case2_1204_soft_low porosity_cyclic/case2_1204_soft_low porosity_cyclic/Air 1",
    "data/bubbles/case2_air1",
)
prepare_case(
    "/home/aaron/Downloads/case3_20250904_90degree_1/case3_20250904_90degree_1/Air1",
    "data/bubbles/case3_air1",
)