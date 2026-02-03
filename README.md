# SafetyNet – Workplace Incident Analytics

SafetyNet is an end-to-end machine learning system designed to analyze
workplace incident narratives and predict the underlying injury cause
and accident category. The project focuses on real-world safety data,
class imbalance handling, and scalable deep learning deployment.

---

## Problem Statement
Workplace incident reports are typically written as unstructured text,
making large-scale safety analysis slow and manual. SafetyNet automates
this process by converting raw narratives into structured accident
categories that can be used for safety audits and preventive planning.

---

## Dataset
- Workplace incident narratives with event titles and descriptions
- Highly imbalanced class distribution (rare but critical incidents)

Key challenges:
- Noisy free-text data
- Overlapping injury contexts (e.g., fractures from falls vs machinery)
- Severe class imbalance

---

## Methodology

### 1. Data Preprocessing & Label Engineering
- Cleaned and filtered incident narratives
- Designed a **hierarchical label grouping strategy** to map granular
  event titles into broader root-cause categories
- Encoded labels using `LabelEncoder`
- Applied **stratified train–validation split** to preserve class ratios

### 2. Handling Class Imbalance
- Computed **class weights** using `sklearn.utils.compute_class_weight`
- Integrated weights into cross-entropy loss to prevent bias toward
  majority classes

### 3. Model Architecture
- Fine-tuned **DistilBERT (distilbert-base-uncased)**
- Used the `[CLS]` token representation for classification
- Added dropout regularization for improved generalization

### 4. Training Strategy
- AdamW optimizer with linear learning-rate scheduling
- Mixed-precision training for GPU efficiency
- Gradient clipping to stabilize training

---

## Results
- Achieved **~81% overall accuracy** on the validation set
- Strong performance on high-frequency injury categories
- Meaningful predictions even for low-support classes after weighting

---

## Deployment
- Model artifacts saved for production inference
- Designed to be deployed on AWS (EC2 + S3)
- Integrated with a React-based dashboard for real-time predictions

---

## Tech Stack
- Python, PyTorch, Hugging Face Transformers
- DistilBERT
- Scikit-learn, Pandas, NumPy
- AWS EC2, S3
- React (frontend)

---

## Author
**Love Chourasia**  
Society of Civil Engineers, IIT Kanpur
