#!/usr/bin/env python3
"""
build_groups.py -- produce groups.csv (columns: name,group) for --groups-csv in nested_cv.py, so
that outer folds don't split frames from the same LIACI inspection video between train and test.

Two phases, run in this order:

  PHASE A (metadata search): looks for anything that already states which video/clip each image
  came from -- a metadata/annotation file shipped with the original LIACI download, or a filename
  pattern that encodes it. This is authoritative if found; prefer it over Phase B.

      python build_groups.py scan --data-root dataset_yolo --liaci-raw /path/to/original/LIACI/download

  PHASE B (visual-similarity fallback, only if Phase A finds nothing usable): LIACI images are
  frames extracted from a small number of source videos, so consecutive frames of the same video
  (by filename order, which for sequential numeric names like image_0005.jpg is presumably close
  to capture order) look almost identical, and there is a visible jump at every cut to a new video.
  This mirrors the approach used in prior work on this same dataset (Islam-style ~0.90 cosine
  similarity cutoff for near-duplicate LIACI frames). It produces a PROVISIONAL grouping: inspect
  the report before trusting it -- this is a heuristic proxy for ground-truth video boundaries, not
  a guarantee.

      python build_groups.py cluster --data-root dataset_yolo --out groups.csv --report groups_report.csv

  Then, once you're satisfied with the report:
      GROUPS_CSV=groups.csv bash run_revision.sh direct       # (etc. -- re-run every phase that trains)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import IMG_EXT, collect_pool  # noqa: E402

METADATA_NAMES = ["metadata.csv", "metadata.json", "annotations.json", "annotations.csv",
                  "video_id.csv", "sequence.csv", "clips.csv", "info.csv", "index.csv"]
METADATA_KEYS = ["video", "video_id", "clip", "clip_id", "sequence", "seq_id", "source_video", "vessel"]


def scan(args):
    print("== Buscando metadata de video/secuencia ==")
    found_any = False
    roots = [Path(args.data_root)]
    if args.liaci_raw:
        roots.append(Path(args.liaci_raw))
    for root in roots:
        if not root.exists():
            print(f"  (no existe: {root})"); continue
        print(f"\n-- explorando {root} --")
        for name in METADATA_NAMES:
            for p in root.rglob(name):
                found_any = True
                print(f"  encontrado: {p}")
                try:
                    if p.suffix == ".json":
                        d = json.load(open(p))
                        keys = list(d[0].keys()) if isinstance(d, list) and d else \
                               (list(d.keys()) if isinstance(d, dict) else [])
                    else:
                        import csv as _csv
                        with open(p) as f:
                            keys = next(_csv.reader(f))
                    hit = [k for k in keys if any(mk in str(k).lower() for mk in METADATA_KEYS)]
                    print(f"    columnas/keys: {keys[:15]}")
                    if hit:
                        print(f"    -> columnas prometedoras: {hit}  (revisa a mano si sirven de 'group')")
                except Exception as e:
                    print(f"    (no se pudo leer: {e})")
        # any other json/csv/txt file that isn't obviously something else
        for ext in ("*.json", "*.csv", "*.txt", "*.xml"):
            for p in root.rglob(ext):
                parts_lower = {x.lower() for x in p.parts}
                if p.name in METADATA_NAMES or "label" in p.name.lower() or parts_lower & {"labels", "images"}:
                    continue
                if p.stat().st_size < 20_000_000:  # skip huge files
                    found_any = True
                    print(f"  otro fichero de metadatos posible: {p}  ({p.stat().st_size/1024:.0f} KB)")
    print("\n== Patron de nombres de archivo ==")
    items = collect_pool(args.data_root)
    names = sorted(i["name"] for i in items)
    print(f"  {len(names)} imagenes. Ejemplos: {names[:3]} ... {names[-3:]}")
    print("  Si el nombre real (o su ruta original en el zip de LIACI) incluye el video/clip de "
          "origen (p.ej. 'ROV3_dive2_frame0123.jpg'), decidmelo y ajusto el parseo del nombre en "
          "vez de usar la Fase B.")
    if not found_any:
        print("\nNo se encontro ningun fichero de metadatos evidente. Si teneis el ZIP/carpeta "
              "ORIGINAL de https://liaci.sintef.cloud/ (antes de convertir a formato YOLO), volved "
              "a ejecutar esto con --liaci-raw apuntando a esa carpeta -- puede contener lo que aqui "
              "falta. Si no existe en ningun sitio, usad la Fase B (comando 'cluster').")


def frame_signature(path, size=24):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def phash(path, hash_size=16, highfreq_factor=4):
    """
    Perceptual hash (DCT-based, like the standard 'phash' algorithm): robust to
    compression/illumination, and -- unlike raw pixel correlation on a tiny thumbnail --
    specifically designed to separate near-duplicate images from genuinely different ones,
    which matters here because underwater hull images from DIFFERENT source videos can share
    a generic look (murky water, hull texture) that fools a crude similarity metric into
    chaining unrelated videos into one group.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img_size = hash_size * highfreq_factor
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    low = dct[:hash_size, :hash_size]
    med = np.median(low)
    return (low > med).ravel()  # boolean hash vector


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def cluster_idgap(args, items):
    """
    Groups by GAPS in the numeric ID embedded in each filename (e.g. 'image_1741.jpg' -> 1741),
    NOT by rank-position in our 261-image subset. This is different from, and often much better
    than, --method sequential: if these numeric IDs are LIACI's own global image index across its
    full ~1893-image release (a common convention for frame-extracted datasets, where IDs are
    assigned in roughly capture/extraction order per source video), then two images with CLOSE
    numeric IDs are likely from the same video even if none of the IDs between them happen to be
    part of our 261-image subset -- whereas --method sequential only looks at rank-adjacency
    within the subset, which carries no such information if the subset is a sparse sample of a
    much larger numbered pool (exactly the situation found for this dataset: filenames range up to
    ~1890+, matching LIACI's documented ~1893 total images, not 1..261).
    Uses the same adaptive local-MAD change-point logic as --method sequential, applied to the
    numeric gap between consecutive (by ID, not by alphabetical rank -- same thing here since IDs
    are zero-padded, but stated for clarity) images instead of a visual-dissimilarity score.
    """
    import re
    def extract_id(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else None
    ids = [extract_id(i["name"]) for i in items]
    if any(x is None for x in ids):
        raise SystemExit("ERROR: no se pudo extraer un numero del nombre de alguna imagen; "
                          "revisa el patron de nombres o usa --method graph/sequential en su lugar.")
    order = sorted(range(len(items)), key=lambda i: ids[i])
    items_sorted = [items[i] for i in order]
    ids_sorted = [ids[i] for i in order]

    gap = [0] + [ids_sorted[i] - ids_sorted[i - 1] for i in range(1, len(ids_sorted))]
    gap = np.array(gap, dtype=float)

    groups_sorted, gid = [0], 0
    window = []
    for i in range(1, len(items_sorted)):
        if args.fixed_cut:
            # Simple global threshold, no local adaptivity at all: cut whenever the gap exceeds
            # --min-id-gap, full stop. Recommended when the printed histogram shows a smooth,
            # unimodal-ish decay with no bimodal "small gap / large gap" separation -- which is
            # what real LIACI data showed us: nothing above 75, no clear jump in the histogram.
            # In that situation a "local baseline" adaptive method (below) has no real structure
            # to adapt to, and in practice over-merged badly (one 111-image group out of 261 in
            # our own test on the real histogram) while still leaving 14 singleton groups.
            is_cut = gap[i] > args.min_id_gap
        elif len(window) < args.min_periods:
            # too few points for a reliable MAD (a 1-2 point MAD is near-zero, which makes the
            # adaptive check hypersensitive to ordinary fluctuation). The absolute floor still
            # applies unconditionally, so a genuine large outlier is still caught even with no
            # local history yet.
            is_cut = gap[i] > args.min_id_gap
        else:
            w = np.array(window[-args.window:])
            med = np.median(w)
            mad = np.median(np.abs(w - med)) + 1e-6
            is_cut = (gap[i] > med + args.k * mad * 1.4826) and gap[i] > args.min_id_gap
        if is_cut:
            gid += 1; window = []
        else:
            window.append(gap[i])
        groups_sorted.append(gid)

    # map back to the ORIGINAL item order (report/CSV keep the usual alphabetical-name order)
    groups = [None] * len(items)
    gap_out = [None] * len(items)
    for pos, orig_idx in enumerate(order):
        groups[orig_idx] = groups_sorted[pos]
        gap_out[orig_idx] = gap[pos]

    all_gaps = gap[1:]
    hist, edges = np.histogram(all_gaps, bins=[0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 100000])
    print("\nHistograma de HUECOS entre IDs consecutivos (todo el dataset, ordenado por numero):")
    for h_, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * min(80, int(h_ / max(1, all_gaps.size) * 400))
        lo = int(e0); hi = "inf" if e1 > 99999 else int(e1)
        print(f"  [{lo:4d}-{str(hi):>4s}] {h_:4d} {bar}")
    return groups, {"id": [str(x) for x in ids], "id_gap_to_prev_by_id": [f"{x:.0f}" for x in gap_out]}


def cluster_sequential(args, items):
    """Original, order-dependent method: kept as an option (--method sequential), but NOT the
    default any more. It assumes filenames are ~chronological within each video; if that
    assumption is false (check with check_filename_order.py first), this method has no usable
    signal and will over-merge or produce nonsense groups regardless of --k/--min-jump."""
    sigs = [frame_signature(i["img"]) for i in items]
    dist = [0.0]
    for i in range(1, len(items)):
        if sigs[i - 1] is None or sigs[i] is None:
            dist.append(1.0); continue
        c = float(np.corrcoef(sigs[i - 1].ravel(), sigs[i].ravel())[0, 1])
        dist.append(1.0 - c)
    dist = np.array(dist)
    groups, gid = [0], 0
    window = []
    for i in range(1, len(items)):
        # same bugfix as cluster_idgap: no free-pass window, absolute floor always applies.
        if len(window) < args.min_periods:
            is_cut = dist[i] > args.min_jump
        else:
            w = np.array(window[-args.window:])
            med = np.median(w)
            mad = np.median(np.abs(w - med)) + 1e-6
            is_cut = (dist[i] > med + args.k * mad * 1.4826) and dist[i] > args.min_jump
        if is_cut:
            gid += 1; window = []
        else:
            window.append(dist[i])
        groups.append(gid)
    return groups, {"jump_to_prev": [f"{d:.4f}" for d in dist]}


def cluster_graph(args, items):
    """
    Default method: order-independent near-duplicate grouping via a perceptual hash (DCT-based)
    and connected components. Does not assume filenames carry any temporal/video information (the
    sequential method above does, and fails silently if that assumption is wrong -- e.g. it produced
    3 groups for 261 real LIACI images, which check_filename_order.py traced to filenames not being
    in capture order and/or generic underwater images not being separable by raw pixel correlation).
    A perceptual hash is deliberately stricter/more structure-sensitive than pixel correlation, to
    avoid chaining unrelated videos together just because they share a generic underwater look.
    Two images are linked if their hash Hamming distance is <= --hamming-thr; groups are the
    connected components of that graph (so A-B and B-C linked still merges A,B,C even if A,C are not
    directly linked -- this transitive "chaining" is intentional for near-duplicate VIDEO FRAMES,
    which form a connected chain frame-by-frame, but is exactly the failure mode to watch for if the
    threshold is too loose; see the histogram this command reports).
    """
    hashes = [phash(i["img"]) for i in items]
    valid = [i for i, h in enumerate(hashes) if h is not None]
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    all_dists = []
    nearest = [None] * n
    for a in range(len(valid)):
        ia = valid[a]
        best = None
        for b in range(len(valid)):
            if a == b:
                continue
            ib = valid[b]
            d = hamming(hashes[ia], hashes[ib])
            all_dists.append(d)
            if best is None or d < best:
                best = d
            if d <= args.hamming_thr:
                union(ia, ib)
        nearest[ia] = best

    roots = [find(i) for i in range(n)]
    remap, groups = {}, []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        groups.append(remap[r])

    all_dists = np.array(all_dists)
    hist, edges = np.histogram(all_dists, bins=list(range(0, 258, 8)))
    print("\nHistograma de distancias de Hamming entre TODAS las parejas de imagenes")
    print("(hash de", hashes[0].size if hashes[0] is not None else "?", "bits; busca un pico bajo separado del resto):")
    for h_, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * min(80, int(h_ / max(1, all_dists.size) * 2000))
        print(f"  [{int(e0):3d}-{int(e1):3d}] {h_:7d} {bar}")
    return groups, {"nearest_hamming": [str(nearest[i]) for i in range(n)]}


def cap_group_sizes(groups, items, max_size):
    """
    Post-process: split any group larger than `max_size` into contiguous-by-name sub-chunks of at
    most `max_size` images each. Needed because a single dominant, densely-sampled sequence can
    otherwise make StratifiedGroupKFold put a hugely disproportionate share of the dataset into one
    outer fold (observed on the real data: a single 111/261-image group, present regardless of
    --method or threshold, produced outer test folds ranging from 3 to 201 images out of 5).
    Splitting a known-dense group is justified here only because a manual check (see the paper's
    Limitations) found that even minimal-ID-gap pairs inside it are NOT near-duplicate frames --
    chunking it does not knowingly reintroduce the leakage this script exists to prevent. If you
    have not done an equivalent check on YOUR data, do not blindly rely on this flag to paper over
    an oversized group: inspect it first (see check_filename_order.py / the printed histogram).
    """
    items_by_group = {}
    for it, g in zip(items, groups):
        items_by_group.setdefault(g, []).append(it["name"])
    remap = {}
    for g, names in items_by_group.items():
        names_sorted = sorted(names)  # contiguous by name == contiguous by numeric id (zero-padded)
        for i, n in enumerate(names_sorted):
            remap[n] = f"{g}_{i // max_size}"
    new_groups_str = [remap[it["name"]] for it in items]
    uniq = {g: i for i, g in enumerate(sorted(set(new_groups_str)))}
    return [uniq[g] for g in new_groups_str]


def cluster(args):
    items = collect_pool(args.data_root)
    items.sort(key=lambda i: i["name"])
    if args.method == "idgap":
        groups, extra_cols = cluster_idgap(args, items)
    elif args.method == "sequential":
        groups, extra_cols = cluster_sequential(args, items)
    else:
        groups, extra_cols = cluster_graph(args, items)

    if args.max_group_size:
        n_before = len(set(groups))
        groups = cap_group_sizes(groups, items, args.max_group_size)
        print(f"\n--max-group-size {args.max_group_size}: {n_before} grupos -> {len(set(groups))} "
              f"tras trocear los que lo superaban.")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "group"])
        for it, g in zip(items, groups):
            w.writerow([it["name"], f"g{g:04d}"])

    extra_keys = list(extra_cols.keys())
    with open(args.report, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "name", "group"] + extra_keys)
        for i, (it, g) in enumerate(zip(items, groups)):
            w.writerow([i, it["name"], f"g{g:04d}"] + [extra_cols[k][i] for k in extra_keys])

    sizes = np.bincount(groups)
    n_groups = len(sizes)
    print(f"\nmetodo={args.method}  {len(items)} imagenes -> {n_groups} grupos")
    print(f"tamano de grupo: min={sizes.min()} mediana={int(np.median(sizes))} max={sizes.max()} media={sizes.mean():.1f}")
    print(f"grupos de 1 sola imagen (huerfanos): {int((sizes==1).sum())} de {n_groups}")
    print(f"\nEscrito: {args.out}  (para nested_cv.py --groups-csv)")
    print(f"Escrito: {args.report}")
    param_hint = {"idgap": "--k / --min-id-gap (o probad --fixed-cut)", "graph": "--hamming-thr",
                  "sequential": "--k / --min-jump"}[args.method]
    if n_groups < 8:
        print(f"\nAVISO: solo {n_groups} grupos para {len(items)} imagenes y ~16-17 videos documentados "
              f"-- probable SOBRE-fusion (chaining). Ajustad {param_hint} (mas estricto) y repetid.")
    elif n_groups > 60:
        print(f"\nAVISO: {n_groups} grupos es mucho mas que los ~16-17 videos documentados -- probable "
              f"SOBRE-fragmentacion. Ajustad {param_hint} (mas laxo) y repetid.")
    if sizes.max() > len(items) / 3:
        print(f"\nNOTA: el grupo mas grande tiene {sizes.max()} de {len(items)} imagenes "
              f"({100*sizes.max()/len(items):.0f}%). Esto NO es necesariamente un error: LIACI puede "
              f"tener videos de duracion/muestreo muy desigual (una inspeccion larga y densamente "
              f"muestreada frente a varias cortas), en cuyo caso un grupo grande es correcto. "
              f"Comprobadlo a ojo: revisad si las imagenes de este grupo (ver {args.report}, columna "
              f"'group') parecen razonablemente la misma inspeccion de principio a fin, en vez de "
              f"asumir que hay que forzar un {param_hint} mas estricto.")
    else:
        print(f"\n{n_groups} grupos es del orden de los ~16-17 videos documentados por LIACI -- "
              f"razonable, pero revisad el report/histograma antes de usarlo para entrenar.")
    print("\nAVISO GENERAL: esto sigue siendo una aproximacion visual, no metadata real de video, y "
          "no ha sido validado contra LIACI real por nosotros. Revisad a mano unas cuantas imagenes "
          "de un mismo grupo grande para confirmar que de verdad parecen el mismo video.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("scan")
    p1.add_argument("--data-root", default="dataset_yolo")
    p1.add_argument("--liaci-raw", default=None)
    p1.set_defaults(func=scan)

    p2 = sub.add_parser("cluster")
    p2.add_argument("--data-root", default="dataset_yolo")
    p2.add_argument("--out", default="groups.csv")
    p2.add_argument("--report", default="groups_report.csv")
    p2.add_argument("--method", choices=["idgap", "graph", "sequential"], default="idgap",
                    help="'idgap' (default): groups by GAPS in the numeric ID embedded in each "
                         "filename (e.g. 'image_1741.jpg' -> 1741), assuming it is LIACI's own global "
                         "image index (filenames observed to range well beyond 261, up to ~1890+, "
                         "matching LIACI's documented ~1893-image release) -- needs no image loading "
                         "at all. 'graph': order-independent near-duplicate grouping via perceptual "
                         "hash + connected components (use if idgap's histogram shows no structure). "
                         "'sequential': the original, rank-position change-point method -- only "
                         "use this if check_filename_order.py confirms filenames carry real temporal "
                         "signal AND idgap is not applicable (e.g. no numeric ID in the filenames).")
    p2.add_argument("--min-id-gap", type=float, default=15.0,
                    help="[idgap method] absolute floor (adaptive mode) or the ONLY threshold (--fixed-cut "
                         "mode): any gap larger than this is a cut.")
    p2.add_argument("--fixed-cut", action="store_true",
                    help="[idgap method] use a single global --min-id-gap threshold instead of the "
                         "adaptive local-MAD method; recommended when the printed histogram shows a "
                         "smooth decay with no clear small-gap/large-gap separation (no local baseline "
                         "to adapt to). Pick --min-id-gap by counting backwards from the histogram's "
                         "tail until you have roughly as many gaps above it as the expected number of "
                         "cuts (documented video count minus 1).")
    p2.add_argument("--hamming-thr", type=int, default=12,
                    help="[graph method] max perceptual-hash Hamming distance (out of 256 bits) to "
                         "link two images as near-duplicates; lower = stricter = more, smaller groups. "
                         "Pick this from the printed histogram: it should sit in the gap between a "
                         "low-distance near-duplicate peak and the rest, if one exists.")
    p2.add_argument("--k", type=float, default=6.0,
                    help="[sequential method] cut when the frame-to-frame jump exceeds (local median + k*MAD)")
    p2.add_argument("--window", type=int, default=15, help="[sequential method] recent-frame window for the local baseline")
    p2.add_argument("--min-jump", type=float, default=0.15,
                    help="[sequential method] absolute floor so near-zero local variance doesn't look like a cut")
    p2.add_argument("--min-periods", type=int, default=3,
                    help="[idgap/sequential methods] minimum points of local history required before "
                         "the adaptive (MAD-based) check activates; below this, only the absolute "
                         "floor (--min-id-gap / --min-jump) is checked. Must be >=2 for the adaptive "
                         "check to be meaningful (MAD of 1 point is always 0).")
    p2.add_argument("--max-group-size", type=int, default=None,
                    help="post-process: split any resulting group larger than this into contiguous "
                         "sub-chunks of at most this many images. Recommended if one group ends up "
                         "dominating the dataset (check the printed group-size distribution first) -- "
                         "an oversized group makes StratifiedGroupKFold produce wildly unbalanced "
                         "outer folds (observed on real data: fold test sizes from 3 to 201 out of "
                         "261, caused by a single 111-image group). Only do this if you have evidence "
                         "the dominant group is not itself a near-duplicate-frame burst (e.g. a manual "
                         "spot check); otherwise you may be re-introducing the leakage this script "
                         "exists to prevent. A sensible starting point is roughly N/k (dataset size "
                         "divided by the number of outer CV folds), e.g. 261/5 ~= 25.")

    p2.set_defaults(func=cluster)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
