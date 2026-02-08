Below is a **complete, copy ready README.md**.
You can paste this **exactly as is**. No changes are required.

---

# ProsthoObjects3D Machine Learning Repository

This repository contains annotation, preprocessing, machine learning, and deep learning code associated with the ProsthoObjects3D three dimensional STL dental dataset. The code supports segmentation, model training, and evaluation for three dimensional prosthetic object analysis on intraoral surface meshes.

---

## Repository structure

```
prosthoobjects3d-ml/
├── annotation/
│   └── segment_and_export_bounding_boxes.py
├── ml/
│   ├── part2_data_preparation.py
│   ├── part3_model_training.py
│   └── part4_model_evaluation.py
├── classical_ml/
│   ├── train_random_forest.py
│   └── train_svm.py
├── README.md
├── LICENSE
└── .gitignore
```

---

## Annotation and segmentation

Bounding boxes are manually placed on standardized intraoral STL meshes using Blender. The script in the `annotation` directory performs Boolean intersection between the full scan mesh and each bounding box to segment individual prosthetic objects. For each object, the script exports a segmented STL file together with a structured JSON annotation containing axis aligned three dimensional bounding box information.

---

## Machine learning and deep learning

The `ml` directory contains scripts implementing a deep learning workflow for three dimensional point cloud analysis. These scripts include data preparation and normalization, definition and training of a PointNet architecture using GPU acceleration, and evaluation of trained models using standard performance metrics.

---

## Classical machine learning baselines

The `classical_ml` directory contains baseline models based on traditional machine learning methods. Support vector machines and random forest classifiers are provided to enable comparison with deep learning based approaches. These models operate on flattened feature representations.

---

## Intended use

This repository is intended to support reproducible research in three dimensional dental computer vision and geometric deep learning. The code is provided for research and benchmarking purposes and is designed to be used in conjunction with the ProsthoObjects3D dataset.

---

## License

This code is released under the MIT License.
