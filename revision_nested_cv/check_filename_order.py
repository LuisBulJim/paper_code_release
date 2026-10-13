#!/usr/bin/env python3
"""
Diagnostico rapido, sin retrain: comprueba si el orden por nombre de fichero lleva
alguna informacion temporal/de video, y si las imagenes llevan fecha en el EXIF.

    python3 check_filename_order.py --data-root dataset_yolo
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="dataset_yolo")
ap.add_argument("--n-samples", type=int, default=40)
a = ap.parse_args()

paths = sorted(p for s in ("train", "val", "test") for p in (Path(a.data_root) / "images" / s).glob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print(f"{len(paths)} imagenes encontradas, ordenadas por nombre.")
print("primeros 5:", [p.name for p in paths[:5]])
print("ultimos 5:", [p.name for p in paths[-5:]])

# --- EXIF check (barato, puede resolver esto de raiz si existe)
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    have_pil = True
except Exception:
    have_pil = False
    print("\nPillow no disponible: `pip install Pillow` para poder comprobar EXIF (opcional, no bloqueante).")

if have_pil:
    n_with_exif = 0
    examples = []
    for p in paths[:200]:
        try:
            exif = Image.open(p).getexif()
            if exif:
                dt = exif.get(306) or exif.get(36867)  # DateTime / DateTimeOriginal
                if dt:
                    n_with_exif += 1
                    if len(examples) < 5:
                        examples.append((p.name, dt))
        except Exception:
            pass
    print(f"\nEXIF con fecha encontrado en {n_with_exif}/200 imagenes de muestra.")
    if examples:
        print("ejemplos:", examples)
        print(">>> Si esto es != 0, dimelo: usar EXIF es mas fiable que similitud visual.")

# --- Adjacent-by-filename vs random-pair similarity check
def sig(p):
    img = cv2.imread(str(p))
    if img is None:
        return None
    img = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel() / 255.0

rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=min(a.n_samples, len(paths)), replace=False)
idx.sort()
sigs = {i: sig(paths[i]) for i in idx}
sigs = {i: s for i, s in sigs.items() if s is not None}
keys = sorted(sigs)

def corr(i, j):
    return float(np.corrcoef(sigs[i], sigs[j])[0, 1])

adjacent = [corr(keys[k], keys[k+1]) for k in range(len(keys)-1)]
rand_pairs = [(int(a_), int(b_)) for a_, b_ in rng.integers(0, len(paths), size=(len(adjacent), 2)) if a_ != b_]
random_sims = []
for a_, b_ in rand_pairs[:len(adjacent)]:
    sa, sb = sig(paths[a_]), sig(paths[b_])
    if sa is not None and sb is not None:
        random_sims.append(float(np.corrcoef(sa, sb)[0, 1]))

adjacent = np.array(adjacent)
random_sims = np.array(random_sims)
print(f"\nSimilitud (correlacion) entre pares CONSECUTIVOS por nombre: media={adjacent.mean():.3f} mediana={np.median(adjacent):.3f}")
print(f"Similitud entre pares ALEATORIOS (sin relacion de orden):    media={random_sims.mean():.3f} mediana={np.median(random_sims):.3f}")
gap = adjacent.mean() - random_sims.mean()
print(f"\nDiferencia (adyacentes - aleatorios) = {gap:+.3f}")
if gap > 0.15:
    print(">>> El orden por nombre SI lleva señal: los adyacentes son claramente mas parecidos que el azar.")
elif gap > 0.05:
    print(">>> Señal debil: el orden por nombre ayuda un poco, pero no es fuerte.")
else:
    print(">>> El orden por nombre NO lleva señal util: los adyacentes se parecen igual que pares al azar.")
    print("    Esto explica por que el agrupamiento por 'salto respecto al vecino' fusiono casi todo:")
    print("    no hay una linea base estable de la que detectar un 'corte'.")
