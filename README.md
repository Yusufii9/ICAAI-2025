# Knowledge-Driven Vision-Language Model for Plexus Detection in Hirschsprung’s Disease
## 📋 Overview

This project proposes a novel framework that integrates expert-derived textual concepts into a CLIP-based vision-language model to guide plexus classification in histopathological whole slide images (WSIs) for Hirschsprung’s Disease (HD).  
Our method aims to combine the strengths of Vision Transformers (ViTs) with clinical knowledge, improving interpretability and alignment with real-world diagnostic processes.

## ✨ Highlights

- **Baseline Model:** Vision Transformer (ViT) fine-tuned on histopathological tiles.
- **Proposed Model:** CLIP-based multi-modal model integrating expert prompts.
- **Dataset:** 30 WSIs from 26 patients (Children’s Hospital of Eastern Ontario). The dataset folder will be left empty, as this is a private dataset and cannot be shared publicly. 
- **Results:**  
  - ViT achieved higher raw classification accuracy (87.17%).  
  - Proposed CLIP model achieved better AUC (91.76%), showing stronger discriminative ability.
- **Contribution:** Demonstrates how expert knowledge can enhance model reasoning in medical imaging tasks.

## 📂 Project Structure
```
KDVLM/
├── ConcepPath/
├── dataset/
├── experiment/plexus_detection/
├── prompts/
├── reports/
├── saved_rp_all/          # Pre-trained features and model checkpoints
├── best_model.pt          # Best trained model weights
├── Fold_1_5_Epoch_Metrics.csv  # Training metrics for cross-validation
├── README.md # Project description and instructions
├── Tutorial.ipynb
├── creating_labels.ipynb
├── report_visuals.ipynb
├── training_setup.ipynb
├── training_setup_v2.ipynb
```

## 🧪 Methodology

- **Data Preprocessing:**
  - Macenko colour normalization
  - Downsampling WSIs to 5× magnification
  - Extraction of 224×224 overlapping tiles
- **Baseline ViT Model:**
  - Fine-tuned using 5-fold cross-validation
- **Proposed CLIP-based Model:**
  - Textual prompts extracted using LLMs (e.g., GPT-4o, DeepSeek-R1)
  - Concept-guided hierarchical aggregation
  - Vision and text encoders from QuiltNet (trained on Quilt-1M dataset)

## 🔥 Results

| Model         | Accuracy (%) | F1-Micro (%) | AUC (%) |
|:--------------|:-------------:|:------------:|:-------:|
| ViT-B16        | 87.17         | 87.17        | 87.17   |
| QuiltNet (CLIP)| 83.93         | 83.93        | 91.76   |

The QuiltNet model showed higher AUC, indicating better overall discriminative power.

## 🏥 Clinical Impact

By aligning AI predictions with expert-driven medical concepts, this method offers enhanced interpretability, better diagnostic consistency, and the potential to aid pathologists in clinical settings.

## 🚀 Future Work

- Expand prompt diversity to improve classification robustness.
- Acquire larger annotated datasets.

## 🤝 Authors

- **Youssef Megahed** — MASc, Data Science, Analytics, and Artificial Intelligence at Carleton University
- **Atallah Madi** — MASc, Electrical and Computer Engineering at Carleton University

📧 Contact: youssefmegahed@cmail.carleton.ca or atallahmadi@cmail.carleton.ca
