Project Overview
This project implements a professional biometric verification pipeline designed for 1:1 Identity Mapping. It leverages deep learning architectures to convert facial features into a digital signature, allowing for highly accurate identity confirmation even in "unconstrained" environments.
The AI Pipeline
The system follows a modular 4-stage architecture to ensure maximum reliability:

Detection & Alignment (MTCNN): A Multi-Task Cascaded CNN identifies 5 facial landmarks (eyes, nose, mouth) to normalize the input.

Robustness Fallback (Manual ROI): I engineered a custom fallback mechanism. If the MTCNN confidence is too low due to environmental factors, the system automatically triggers a Manual Region of Interest (ROI) crop to maintain system availability.

Feature Extraction (FaceNet): The aligned face is mapped into a 128-dimensional embedding space using the InceptionResnetV1 model.

Verification (Euclidean Distance): Identity is confirmed by calculating the L2 distance between embeddings. A distance below the industrial threshold of 0.7 confirms a match.

Technical Results
Dataset: LFW (Labeled Faces in the Wild).

Hardware Optimization: CPU-only inference using .eval() and torch.no_grad() to prevent thermal throttling on portable devices.

Experimental Result: Achieved a Similarity Score of 0.0019 on target identity (George W Bush), representing near-perfect mathematical confidence.
