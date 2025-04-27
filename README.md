# CSI5386 Project — Aligning Knowledge Concepts to Whole Slide Images for Hirschsprung’s Disease Classification

## 📋 Overview

This project proposes a novel framework that integrates expert-derived textual concepts into a CLIP-based vision-language model to guide plexus classification in histopathological whole slide images (WSIs) for Hirschsprung’s Disease (HD).  
Our method aims to combine the strengths of Vision Transformers (ViTs) with clinical knowledge, improving interpretability and alignment with real-world diagnostic processes.

## ✨ Highlights

- **Baseline Model:** Vision Transformer (ViT) fine-tuned on histopathological tiles.
- **Proposed Model:** CLIP-based multi-modal model integrating expert prompts.
- **Dataset:** 30 WSIs from 26 patients (Children’s Hospital of Eastern Ontario).
- **Results:**  
  - ViT achieved higher raw classification accuracy (87.17%).  
  - Proposed CLIP model achieved better AUC (91.76%), showing stronger discriminative ability.
- **Contribution:** Demonstrates how expert knowledge can enhance model reasoning in medical imaging tasks.

## 📂 Project Structure

CSI5386---NLP/ ├── ConcepPath/ # Concept learning components ├── dataset/ # Placeholder for dataset structure ├── experiment/plexus_detection/ # Input and Output saved files ├── prompts/ # Prompt files used for CLIP model ├── reports/ # Analysis reports and visualizations ├── saved_rp_all/dataset_quilt1m_5x_224/ # Precomputed dataset features ├── Fold_1_5_Epoch_Metrics.csv # Training metrics for cross-validation ├── README.md # Project description and instructions ├── Tutorial.ipynb # Tutorial and demonstration notebook ├── creating_labels.ipynb # Label generation scripts ├── report_visuals.ipynb # Visualizing model outputs ├── training_setup.ipynb # Initial training configuration ├── training_setup_v2.ipynb # Updated training configuration


## 🧪 Methodology

- **Data Preprocessing:**
  - Macenko color normalization
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
| Quilt-1M (CLIP)| 83.93         | 83.93        | 91.76   |

The Quilt-1M model showed higher AUC, indicating better overall discriminative power.

## 🏥 Clinical Impact

By aligning AI predictions with expert-driven medical concepts, this method offers enhanced interpretability, better diagnostic consistency, and potential to aid pathologists in clinical settings.

## 🚀 Future Work

- Expand prompt diversity to improve classification robustness.
- Acquire larger annotated datasets.
- Optimize computational efficiency for real-time applications.
- Conduct clinical validation studies.

## 🤝 Authors

- **Youssef Megahed** — Carleton University
- **Atallah Madi** — Carleton University
- **Rowan Hussein** — University of Ottawa

📧 Contact: youssefmegahed@cmail.carleton.ca
