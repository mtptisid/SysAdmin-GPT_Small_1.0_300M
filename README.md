# SysAdmin-GPT: AI-Powered Linux System Management

## BERT-Based Classification Model for Red Hat Documentation

## Overview
This repository contains a **BERT-based sequence classification model** trained to classify queries related to Red Hat Enterprise Linux (RHEL) documentation. The model is fine-tuned on domain-specific queries and provides category-based predictions for better information retrieval and automated assistance.

## Model Details
- **Model Type:** BERT (Bidirectional Encoder Representations from Transformers)
- **Base Model:** `bert-base-uncased`
- **Task:** Text Classification
- **Pretrained on:** General English text (via `bert-base-uncased`)
- **Fine-tuned on:** Red Hat-related queries
- **Number of Classes:** Multiple categories based on documentation topics (e.g., `Security`, `Networking`, `Storage`)
- **Output:** Predicted class label with confidence score

## Features
✔️ **Domain-Specific Fine-Tuning:** Trained on RHEL-related queries for precise classification.  
✔️ **High Accuracy & Performance:** Utilizes Transformer-based architecture for robust text understanding.  
✔️ **Scalability:** Can be extended for broader Linux documentation or IT support queries.  
✔️ **Deployment-Ready:** Model can be integrated into AI-powered documentation search systems or chatbots.

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/redhat-docs-llm.git
cd redhat-docs-llm
```

### 2. Install Dependencies
```bash
pip install transformers torch
```

### 3. Load Model and Tokenizer
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("/content/redhat-docs-llm")
model.eval()
```

## Usage
### Run Inference
```python
# Sample input text
text = "How do I configure a firewall in RHEL?"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Extract class probabilities
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()
confidence = torch.softmax(logits, dim=1)[0, predicted_class].item()

# Display result
print(f"Predicted Category: {predicted_class}\nConfidence: {confidence:.4f}")
```

## Model Training Process
1. **Dataset Preparation**: Collected domain-specific queries and labeled them into predefined categories.
2. **Preprocessing**: Tokenized text using the `bert-base-uncased` tokenizer.
3. **Fine-Tuning**: Trained BERT with a classification head on the labeled dataset.
4. **Evaluation**: Assessed performance using accuracy, precision, and recall metrics.
5. **Deployment**: Converted and saved the model for inference.

## Possible Improvements
🔹 **Use a Larger Model:** Consider using `roberta-large` or `BERTweet` for better contextual understanding.  
🔹 **Increase Training Data:** Expand the dataset with more labeled Red Hat documentation queries.  
🔹 **Fine-Tune for Question Answering:** Instead of classification, train a QA model like `T5` for direct answers.  
🔹 **Optimize Deployment:** Convert model to ONNX or TensorRT for faster inference in production.

## Applications
✅ **Automated Documentation Assistance**: Helps users find relevant Red Hat documentation faster.  
✅ **IT Support Chatbots**: Integrates with AI-powered chatbots to classify user queries.  
✅ **Enterprise Knowledge Bases**: Enhances search functionalities in internal documentation systems.  
✅ **Security & Compliance Monitoring**: Identifies security-related queries for proactive support.


---

## Contact
For any queries, reach out to **[msidrm455@gmail.com](mailto:msidrm455@gmail.com)** or open an issue in the repository.

---

