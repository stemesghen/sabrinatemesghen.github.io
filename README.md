Personal website of Sabrina Temesghen, an AI/ML Engineer with a background in Molecular Science and Software Engineering from UC Berkeley. I build machine learning, generative AI, and data-driven systems with applications across healthcare, biology, and data science. Explore my projects below.

# Portfolio Projects

## Table of Contents

1. [Hybrid RAG System for Document Question Answering](#hybrid-rag-system-for-document-question-answering)
2. [Telematics Usage-Based Insurance (UBI) Pipeline](#telematics-usage-based-insurance-ubi-pipeline)
3. [Bringing Functional Dark Matter Annotations into the Fold](#bringing-functional-dark-matter-annotations-into-the-fold)
4. [Analytic Gradient Implementation for CNDO/2 Hartree–Fock](#analytic-gradient-implementation-for-cndo2-hartree–fock)
5. [3D MRI Tumor Segmentation and Classification](#3d-mri-tumor-segmentation-and-classification)
6. [Disaster Type and Damage Level Classification from Satellite Imagery](#disaster-type-and-damage-level-classification-from-satellite-imagery)
7. [Anomaly Detection for Parasitized Cell Images](#anomaly-detection-for-parasitized-cell-images)
8. [Artificial Neural Network for Regression Tasks](#artificial-neural-network-for-regression-tasks)
9. [Surrogate Modeling with Lasso and Ridge Regression](#surrogate-modeling-with-lasso-and-ridge-regression)
10. [Monte Carlo Simulation for Lennard-Jones Potential](#monte-carlo-simulation-for-lennard-jones-potential)

---

## Hybrid RAG System for Document Question Answering | Python, Elasticsearch, Docling, Qwen, Sentence Transformers, Gemini

[GitHub Repo](https://github.com/stemesghen/torch-ai-rag-assessment.git)

Built an end-to-end retrieval-augmented generation (RAG) system for grounded question answering over complex documents. Parsed and structure-aware chunked source documents with Docling, generated 1024-dimensional dense embeddings using Qwen, and indexed chunk text, embeddings, and document metadata in Elasticsearch. Implemented hybrid retrieval combining BM25 lexical search with kNN dense vector search, followed by Reciprocal Rank Fusion (RRF) and cross-encoder reranking to select the most relevant context for generation.

Preserved document provenance throughout the retrieval pipeline, including source document, page, section heading, and chunk identifiers, enabling generated answers to provide traceable source citations. Integrated Gemini for structured answer generation with explicit insufficient-context behavior for unsupported questions. Evaluated retrieval quality using Recall@K, nDCG, and MRR alongside negative-query testing to assess groundedness and reduce unsupported generation.

The system demonstrates an end-to-end workflow spanning document ingestion, metadata-aware indexing, dense and sparse retrieval, rank fusion, reranking, LLM generation, evaluation, and source attribution.

---
---

## Telematics Usage-Based Insurance (UBI) Pipeline | Python, scikit-learn, FastAPI, Docker  
[GitHub Repo](https://github.com/stemesghen/telematics-integration-auto-insurance-project.git)

Built an end-to-end proof-of-concept pipeline for usage-based auto insurance, simulating trip-level driving data and aggregating into monthly driver features. Trained a calibrated logistic regression risk model and mapped predictions to capped pricing factors (0.9–1.1) for fairness and regulatory alignment. Exposed RESTful APIs (`/score`, `/pricing`) via FastAPI with an optional Streamlit dashboard for interactive exploration. Containerized the pipeline with Docker and Docker Compose for reproducibility and scalable deployment. Achieved >90% calibration accuracy on simulated data, balancing interpretability, risk fairness, and deployment readiness.

---

## Bringing Functional Dark Matter Annotations into the Fold | Python, FoldSeek, MMseqs2  
[GitHub Repo](https://github.com/JXDolan/bring_dark_matter_into_fold)

Designed a structure-aware protein annotation pipeline to detect remote homologs with low sequence similarity using predicted protein folds. Integrated ESMFold and FoldSeek with MetaPathways to identify structurally similar proteins from the ESMAtlas database, filtering for high TM-align scores. Developed a hybrid annotation database by combining canonical sequences (SwissProt) with structural homologs identified via FoldSeek, improving recall and coverage of difficult-to-detect functions. Used MMseqs2 to validate the augmented database on metagenome-assembled genomes (MAGs) containing both SQR-positive and SQR-negative controls. Conducted parameter sweeps (TM-score, E-value, coverage) over 486 combinations to optimize annotation precision and recall, achieving 52 combinations with perfect performance. Addressed infrastructure limitations by automating access to large structure databases via FoldSeek’s web API and pipeline integration of ESMFold for structure prediction from sequence.

---

## Analytic Gradient Implementation for CNDO/2 Hartree–Fock | C++, Armadillo  
[GitHub Repo](https://github.com/stemesghen/Numerical-Algorithms-Applied-to-Computational-Quantum-Chemistry/tree/main)

Engineered an analytic nuclear gradient module into a CNDO/2 Self-Consistent Field (SCF) quantum chemistry engine to enable efficient geometry optimization without relying on finite-difference derivatives. This involved deriving and implementing the x and y coefficient matrices from SCF energy expressions, coding the derivatives of contracted Gaussian overlap integrals (s and p functions) with respect to nuclear coordinates, and computing γ<sub>AB</sub> integral derivatives while leveraging translational invariance and symmetry to reduce redundant calculations by 50%. I also implemented the derivative of the nuclear repulsion term (V<sub>nuc</sub>) and assembled the full 3N-dimensional gradient vector for all atoms. Performance optimizations achieved O(N²) scaling for large systems, and results were validated against finite-difference gradients with errors below 10⁻⁶ Hartree/Bohr. The implementation was tested on diatomic and triatomic molecules (CO, HF, H₂O, NH₃), with optimized geometries matching reference data from Pople & Beveridge.

---

## 3D MRI Tumor Segmentation and Classification | Python, TensorFlow, Keras  
[GitHub Repo](https://github.com/stemesghen/3D-MRI-Tumor-Segmentation-and-Classification-3D-Unet-Model-.git)

Developed a 3D U-Net model for binary classification and segmentation of brain tumors using BraTS2020 MRI datasets. Preprocessed multimodal MRI scans (T1CE, T2-weighted, T2-FLAIR) by scaling with MinMaxScaler, combining channels, cropping black spaces, and filtering underrepresented labels. Implemented data augmentation using patchify to create smaller patches (64×64×64) for computational efficiency and balanced training. Trained and validated the model on 369 patient scans (augmented to 738) using custom loss functions (Dice Coefficient, Dice Loss, IoU), batch tuning, and hyperparameter optimization; evaluated model performance across accuracy (up to 97.82%), Dice (0.7130–0.9307), and IoU (0.5617–0.8851) metrics. Visualized segmentation outputs and performance trends using Matplotlib and conducted error analysis to identify false positives and missed tumor regions.

---

## Disaster Type and Damage Level Classification from Satellite Imagery | Python, scikit-learn, OpenCV, TensorFlow  
[GitHub Repo](https://github.com/stemesghen/Natural-Disaster-Detection---Computer-Vision.git)

Integrated machine learning pipeline to classify disaster types and damage levels from over 26,000 satellite images covering Hurricane Matthew, Southern California fires, and Midwest flooding. Implemented preprocessing techniques including resizing, pixel normalization, and thresholding. Extracted features with Sobel edge detection, RGB channel statistics, Local Binary Patterns (LBP), and Gabor filters. Trained logistic regression and random forest models for binary (disaster type) and multiclass (damage level) classification. Tuned hyperparameters using grid search with cross-validation, achieving over 93% accuracy in disaster type classification. Addressed class imbalance with sampling strategies and class weighting, and analyzed confusion matrices to interpret bias and misclassification patterns.

---

## Anomaly Detection for Parasitized Cell Images | Python, TensorFlow, Keras  
[GitHub Repo](https://github.com/stemesghen/Anomaly-Detection-for-Parasitized-Cell-Images-.git)

Applied a convolutional autoencoder for anomaly detection in cell images. Preprocessed images by resizing, thresholding, and normalizing for computational efficiency. Built an encoder-decoder architecture with a latent bottleneck layer to identify parasitized cells via reconstruction errors. Calculated reconstruction errors and latent-space densities using Kernel Density Estimation to classify anomalies. Achieved effective separation between healthy and parasitized samples with a mean reconstruction error ratio of 3:1.

---

## Artificial Neural Network for Regression Tasks | Python, Keras, TensorFlow  
[GitHub Repo](https://github.com/stemesghen/Artificial_Neural_Network_model.git)

Trained a fully connected ANN for non-linear function approximation. Constructed a multi-layer feedforward architecture with ReLU activations and MSE loss. Normalized input features and implemented early stopping and batch training to optimize learning efficiency. Compared ANN performance against Lasso and Ridge regression models, showing improved fit in high-dimensional, non-linear cases. Visualized training loss and prediction accuracy to assess convergence and generalization.

---

## Surrogate Modeling with Lasso and Ridge Regression | Python, scikit-learn  
[GitHub Repo](https://github.com/stemesghen/Lasso_and_Ridge_Regression_Model.git)

Constructed surrogate models using Lasso and Ridge regression to approximate an unknown target function. Designed polynomial feature matrices and performed hyperparameter optimization via grid search to tune regularization strength (alpha). Evaluated model performance using R² scores with both train-test split and the entire dataset. Demonstrated differences in behavior: Lasso produced sparse solutions via feature selection, while Ridge improved generalization by mitigating multicollinearity.

---

## Monte Carlo Simulation for Lennard-Jones Potential | C++, Python, STL  
[GitHub Repo](https://github.com/stemesghen/Monte-Carlo-Simulation.git)

Implemented a Monte Carlo simulation in both Python and C++ to study particle systems interacting via the Lennard-Jones potential. Developed features such as periodic boundary conditions, tail corrections, and cubic lattice generation for initial configurations. Optimized the C++ implementation using STL for efficient memory management and leveraged Python for rapid prototyping and testing. Incorporated acceptance-rejection criteria and long-range interaction corrections to enhance simulation accuracy. Benchmarked performance between Python and C++ implementations, analyzing trade-offs between computational efficiency and development speed.
