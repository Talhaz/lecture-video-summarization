# Lecture Video Summarization using Deep Learning

## Project Overview
Automated lecture video summarization using HBBEA-optimized Deep Residual Network.

## Author
**Mobina Abdul Rauf** (CMS ID: 400944)

### Committee Members
- Dr. Farzana Jabeen
- Dr. Bilal Ali

### Supervisor
- Dr. Shah Khalid

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd lecture_video_summarization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup dataset
python scripts/dataset_downloader.py
```

## Project Structure

```
lecture_video_summarization/
├── data/                  # Data directory
│   ├── raw_videos/       # Raw lecture videos
│   ├── processed/        # Processed data
│   └── features/         # Extracted features
├── src/                  # Source code
│   ├── preprocessing/    # Video preprocessing
│   ├── feature_extraction/  # Feature extraction
│   ├── optimization/     # HBBEA optimization
│   └── models/          # DRN model
├── outputs/              # Output files
├── configs/              # Configuration files
└── notebooks/            # Jupyter notebooks
```

## Usage

### 1. Download Dataset
```bash
python scripts/dataset_downloader.py
```

### 2. Load and Verify Dataset
```bash
python src/utils/video_loader.py
```

### 3. Preprocess Videos
```bash
python src/preprocessing/video_shot_segmentation.py
```

### 4. Extract Features
```bash
python src/feature_extraction/extract_all_features.py
```

### 5. Train Model
```bash
python src/models/train_drn.py
```

## Methodology

1. **Video Shot Segmentation** - YCbCr color space model
2. **Audio-Video Segmentation** - HBBEA optimization
3. **Feature Extraction**
   - Audio: BFCC, MFCC, spectral features
   - Video: SLBT, LTP, HoG, LOOP, LVP
4. **Deep Learning** - HBBEA-optimized DRN
5. **Summarization** - Merge important segments

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- NPV (Negative Predictive Value)

## References
- Kaur, P.C., & Ragha, L. (2024). Optimized deep learning enabled lecture audio video summarization. 
  Journal of Visual Communication and Image Representation, 104, 104309.

## License
Academic use only
