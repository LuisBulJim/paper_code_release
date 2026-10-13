# Project page (`docs/`)

Static, self-contained project page of the revised manuscript **"Underwater Corrosion Detection with YOLO26:
Comparing Model Scaling, Ensemble Fusion, and Test-Time Augmentation"**. It is served by GitHub Pages from this
folder: <https://luisbuljim.github.io/paper_code_release/>.

* `index.html` — the page (all CSS/JS inline, no external dependencies; opens offline too).
* `.nojekyll` — GitHub serves the raw HTML as-is.

The page quotes only numbers that exist in `results/` (abstract, key-result cards and table). The qualitative
image gallery of the first submission (YOLO26 + SAM 3 triptychs produced by the superseded notebook 09) was
removed; the images are kept under `legacy/original_submission/docs_assets/`.

## Publishing

Settings → Pages → Build and deployment → Source: *Deploy from a branch*, Branch `main`, Folder `/docs`.
After a push to `main` the site rebuilds within a few minutes.

## Placeholders still to fill

* "Archived release (DOI) — to be added" button, the DOI in the Data availability paragraph and in the BibTeX
  `note` field (after the Zenodo deposit; see `RELEASE_CHECKLIST.md`).
* The paper button and the `journal` field once the article is published.
