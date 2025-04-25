# TotalDefense
Link to dataset is
https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main

 
Total Defense: A Unified Framework for Securing Continual Learning Against BrainWash Attacks
Total Defense is a robust, modular defense framework designed to protect continual learning (CL) models from advanced data poisoning attacks such as BrainWash and BadNets. Continual learners are especially vulnerable to semantic backdoors and task-specific forgetting, and Total Defense addresses this gap with a three-layered approach that requires no architectural changes and introduces minimal overhead.

How It Works
Total Defense operates through three complementary components:

Data Sanitization Layer
Filters poisoned samples before they influence learning.
Uses statistical outlier detection (Z-scores), spectral anomaly detection (Fourier-based), and adaptive thresholding to flag malicious data.

Model Integrity Layer
Protects the model’s trustworthiness with secure SHA-256 checkpoint hashes.
Automatically reverts to the last verified state on integrity failure using versioned backups and task metadata.

Runtime Mitigation Layer
Defends against residual poison during training.
Applies parameter smoothing, adaptive gradient clipping based on Fisher Information, and sharpness monitoring to identify instability in the loss landscape.

Why It Works
Total Defense is attack-agnostic, lightweight, and compatible with popular CL algorithms like EWC, MAS, and replay-based strategies. It dynamically adapts its thresholds and performs both proactive and reactive interventions to prevent catastrophic forgetting—even in stealthy attack scenarios.

Key Features
Protects against BrainWash-style and semantic backdoor attacks

No changes to core CL architecture required

Less than 5% runtime overhead

Built-in anomaly and hash-based rollback detection

Up to 92% reduction in attack success rate (validated on CIFAR-100 and miniImageNet)