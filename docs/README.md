# Project page (`docs/`)

Static, self-contained project page for the manuscript
**"Underwater Corrosion Detection via YOLO26: From Heterogeneous Ensembles to Robust
Inference with Test-Time Augmentation."**

## Publish with GitHub Pages

1. Push this repository (the page lives in `docs/`).
2. On GitHub: **Settings → Pages → Build and deployment → Source: _Deploy from a branch_**,
   Branch: `main`, Folder: **`/docs`**. Save.
3. The site goes live at `https://<username>.github.io/paper_code_release/`.

`.nojekyll` is present so GitHub serves the raw `index.html` as-is (no Jekyll processing).
The page has **no external dependencies** (all CSS/JS inline) — it also works offline by
opening `index.html` in a browser.

## Add the result images

The page shows images that are **not tracked yet** — they are produced by
`make_paper_figures.py` (run on the server). Missing images auto-hide, so the page looks
clean even before you add them. Drop the selected PNGs into `docs/assets/` with these names:

| File in `assets/` | Source (from `outputs/`) | Used in |
|---|---|---|
| `teaser.png`   | a striking `det_seg/*.png` | hero teaser |
| `gallery1.png` | `det_seg/*.png` | Proposal gallery |
| `gallery2.png` | `det_seg/*.png` | Proposal gallery |
| `gallery3.png` | `det_seg/*.png` | Proposal gallery |
| `gallery4.png` | `det_seg/*.png` | Proposal gallery |

## Before going public — fill the placeholders in `index.html`

- Author list / affiliations (`<!-- TODO -->` in the hero).
- BibTeX `author` field (and journal once accepted).
- Confirm the repository URL/casing in the buttons and links.
