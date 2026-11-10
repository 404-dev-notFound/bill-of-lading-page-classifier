# Bill of Lading Page Classifier

A CNN that decides which pages of a shipping PDF are worth sending to data extraction. Logistics PDFs often bundle the House/Master Bill of Lading (HBL/MBL) with cover sheets, terms and conditions, and blank or scanned filler pages. Classifying pages first means the expensive OCR + LLM extraction step only runs on pages that matter.

This is an early prototype of the page-classification stage from my logistics document-intelligence work, where the production classifier reached 95.9% precision and cut downstream processing cost by about 35%.

## How it works

1. **Render pages:** `pdf_to_image.py` uses PyMuPDF to render every page of every PDF in a folder to PNG at 2× zoom.
2. **Classify pages:** `pdf_page_classifier.py` trains a binary CNN (`good` = relevant, `bad` = irrelevant) on 224×224 page images.
   - **Models:** a `full` 4-block CNN (Conv-BN-Pool-Dropout, global average pooling, dense head) and a `lightweight` variant for CPU inference.
   - **Transfer-learning option:** ResNet50 or MobileNetV2 backbones.
   - **Training:** augmentation (rotation, shift, shear, zoom, brightness), Adam, early stopping, LR-on-plateau, best-checkpoint saving.
   - **Metrics:** accuracy, precision, recall. Precision matters most, because a wrongly dropped page means missing data.

## Data layout

```
data/
├── train/       good/  bad/
├── validation/  good/  bad/
└── test/        good/  bad/
```
`create_directory_structure()` creates this layout. No training data is included (the original documents are confidential).

## Usage

```bash
pip install tensorflow pymupdf pillow numpy
python pdf_to_image.py            # set pdf_folder_path / output_folder_path first
```

```python
from pdf_page_classifier import PDFPageClassifier

clf = PDFPageClassifier(model_type="lightweight")
clf.create_model()
train_gen, val_gen = clf.create_data_generators("data/train", "data/validation")
clf.train(train_gen, val_gen, epochs=50)

label, confidence, prob = clf.predict("page_1.png")
```

## Stack
Python, TensorFlow/Keras, PyMuPDF, Pillow, NumPy
