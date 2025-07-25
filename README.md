# sabrinatemesghen.github.io
Personal website of Sabrina Temesghen, a Software Engineer and Machine Learning Developer with a background in Molecular Science and Software Engineering from UC Berkeley. I build production-ready tools that blend AI with real-world impact in biology, healthcare, and data science. Explore my projects below.



Projects:

Bringing Functional Dark Matter Annotations into the fold| Python, Foldseek, MMseq2		   Jan. 2025 - May 2025
https://github.com/JXDolan/bring_dark_matter_into_fold
Designed and implemented a structure-aware protein annotation pipeline to detect remote homologs with low sequence similarity using predicted protein folds.
Integrated ESMFold and FoldSeek with MetaPathways to identify structurally similar proteins from the ESMAtlas database, filtering for high TM-align scores.
Developed a hybrid annotation database by combining canonical sequences (SwissProt) with structural homologs identified via FoldSeek, improving recall and coverage of difficult-to-detect functions.
Used MMseqs2 to validate the augmented database on metagenome-assembled genomes (MAGs) containing both SQR-positive and SQR-negative controls.
Conducted parameter sweeps (TM-score, E-value, coverage) over 486 combinations to optimize annotation precision and recall, achieving 52 combinations with perfect performance.
Addressed infrastructure limitations by automating access to large structure databases via FoldSeek’s web API and pipeline integration of ESMFold for structure prediction from sequence.


3D MRI Tumor Segmentation and Classification | Python, TensorFlow, Keras		  Nov. 2024  - Dec. 2024
https://github.com/stemesghen/3D-MRI-Tumor-Segmentation-and-Classification-3D-Unet-Model-.git
Designed and implemented a structure-aware protein annotation pipeline to detect remote homologs with low sequence similarity using predicted protein folds.
Integrated ESMFold and FoldSeek with MetaPathways to identify structurally similar proteins from the ESMAtlas database, filtering for high TM-align scores.
Developed a hybrid annotation database by combining canonical sequences (SwissProt) with structural homologs identified via FoldSeek, improving recall and coverage of difficult-to-detect functions.
Used MMseqs2 to validate the augmented database on metagenome-assembled genomes (MAGs) containing both SQR-positive and SQR-negative controls.
Conducted parameter sweeps (TM-score, E-value, coverage) over 486 combinations to optimize annotation precision and recall, achieving 52 combinations with perfect performance.
Addressed infrastructure limitations by automating access to large structure databases via FoldSeek’s web API and pipeline integration of ESMFold for structure prediction from sequence.


Anomaly Detection for Parasitized Cell Images | Python, TensorFlow, Keras, Machine Learning                                            Nov. 2024 
https://github.com/stemesghen/Anomaly-Detection-for-Parasitized-Cell-Images-.git
Designed and implemented a convolutional auto encoder for anomaly detection in cell images using Keras and TensorFlow.
Preprocessed images by resizing, thresholding, and normalizing, ensuring computational efficiency and reduced input dimensionality.
Built an encoder-decoder architecture with a latent bottleneck layer to identify parasitized cells via reconstruction errors.
Calculated reconstruction errors and latent-space densities using Kernel Density Estimation to classify anomalies.
Evaluated the model’s performance, achieving effective separation between healthy and parasitized samples with a mean reconstruction error ratio of 3:1.


Artificial Neural Network for Regression Tasks | Python, Keras, TensorFlow, Machine Learning                                           Oct. 2024
https://github.com/stemesghen/Artificial_Neural_Network_model.git
Designed and trained a fully connected artificial neural network (ANN) for non-linear function approximation. Constructed a multi-layer feedforward architecture using Keras with ReLU activations and MSE loss. Normalized input features and implemented early stopping and batch training to optimize learning efficiency. Compared ANN performance against traditional regression models (Lasso and Ridge), showing improved fit in high-dimensional, non-linear cases. Visualized training loss and prediction accuracy to assess convergence and generalization.


Surrogate Modeling with Lasso and Ridge Regression | Python, scikit-learn                                                               Oct. 2024
https://github.com/stemesghen/Lasso_and_Ridge_Regression_Model.git
Constructed surrogate models using Lasso and Ridge regression to approximate an unknown target function. Designed polynomial feature matrices and performed hyperparameter optimization via grid search to tune regularization strength (alpha). Evaluated model performance using both train-test split and the entire dataset, comparing predictive accuracy using R² scores. Demonstrated differences in model behavior, with Lasso yielding sparse solutions via feature selection and Ridge improving generalization by mitigating multicollinearity.


Monte Carlo Simulation for Lennard-Jones Potential | C++, Python, STL, Physics Simulation	                                              Aug. 2024
https://github.com/stemesghen/Monte-Carlo-Simulation.git
Implemented a Monte Carlo simulation in both Python and C++ to study particle systems interacting via the Lennard Jones potential.
Developed features such as periodic boundary conditions, tail corrections, and cubic lattice generation for initial configurations.
Optimized the C++ implementation using STL for efficient memory management and leveraged Python for rapid prototyping and testing. 
Incorporated acceptance-rejection criteria and long-range interaction corrections to enhance simulation accuracy in both implementations. 
Benchmarked performance between Python and C++ implementations, analyzing trade-offs between computational efficiency and development speed

