# Med Vision AI - Backend

AI-based digital prescription OCR + patient medical history system.
This is the backend, starting with the OCR core (highest priority: accuracy).

## Structure

```
med_vision_ai/
├── preprocessing/
│   └── pipeline.py       # Image cleanup: deskew, denoise, CLAHE, adaptive binarize
├── ocr/                  # (next) TrOCR fine-tuning + inference
├── backend/              # (later) FastAPI app - upload, verify, retrieve endpoints
├── tests/
│   └── test_preprocessing.py   # Visual test with a synthetic messy prescription
└── requirements.txt
```

## Status

- [x] **Preprocessing pipeline** - deskew, denoise, CLAHE contrast, adaptive binarize
- [ ] TrOCR baseline inference test
- [ ] Stage A fine-tuning (IAM Handwriting Database)
- [ ] Stage B fine-tuning (RxHandBD - prescription-specific)
- [ ] Medicine reference DB + RapidFuzz fuzzy matching
- [ ] Confidence scoring (OCR score + match score combined)
- [ ] FastAPI backend (upload → OCR → match → verify → store)
- [ ] Doctor correction logging (foundation for the self-learning loop)

## Running what exists now

```bash
pip install -r requirements.txt
python tests/test_preprocessing.py
```

This generates a synthetic messy prescription image (tilted, noisy, uneven
lighting) and runs it through the full preprocessing pipeline, saving a
side-by-side comparison to `/tmp/preprocessing_debug_grid.png` so you can
visually verify each stage is working.

**Once you have real prescription images** (RxHandBD or your own photos),
use `preprocess_file("path/to/image.jpg")` directly - no other code changes
needed. Test on a handful of your messiest real photos (worst lighting,
most tilt) before moving to the OCR model - if preprocessing struggles here,
it will silently hurt every downstream accuracy number.

## Design notes worth remembering

- **`PreprocessConfig` is deliberately separate from the logic.** When you
  start fine-tuning the OCR model, whatever config values you use here must
  be applied identically at training time and inference time. Import this
  same config object in both places - don't hand-retype values.
- **Skew detection is approximate** (see `detect_skew_angle`). It's fast and
  dependency-free, but under-corrects on sparse/curvy input. If real RxHandBD
  images show poor deskewing, the fix is a Hough-transform-based line
  detector - swap it into `detect_skew_angle` without touching anything else.
- **`save_debug_grid()` is your debugging tool for the rest of this project.**
  Any time OCR accuracy looks wrong on a specific image, run it through this
  first to see which preprocessing stage is helping or hurting.

## Next step

TrOCR baseline inference test (no fine-tuning yet) - confirms the model
loads and runs on your GPU before we invest time in the fine-tuning stages.
