import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

INTENT_DATA = [
    {"text": "where is my order", "label": "order_status"},
    {"text": "I want to cancel my order", "label": "cancel_order"},
    {"text": "give me a refund", "label": "refund"},
    {"text": "payment didn't go through", "label": "payment_issue"},
    {"text": "shipping is taking too long", "label": "shipping_delay"},
    {"text": "I received a wrong item", "label": "wrong_item"},
    {"text": "change my delivery address", "label": "change_address"},
    {"text": "track my package", "label": "track_order"},
    {"text": "tell me about the product", "label": "product_info"},
    {"text": "what is your return policy", "label": "return_policy"},
    {"text": "I want to complain", "label": "complaint"},
    {"text": "hello bot", "label": "greeting"},
]

INTENT_LABELS = sorted(set(d["label"] for d in INTENT_DATA))
intent2id = {l:i for i,l in enumerate(INTENT_LABELS)}

texts = [d["text"] for d in INTENT_DATA]
labels = [intent2id[d["label"]] for d in INTENT_DATA]

# split
random.seed(42)
idx = list(range(len(texts)))
random.shuffle(idx)
split = int(0.8*len(idx))
train_idx, val_idx = idx[:split], idx[split:]

X_train = [texts[i] for i in train_idx]
X_val = [texts[i] for i in val_idx]
y_train = [labels[i] for i in train_idx]
y_val = [labels[i] for i in val_idx]

logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

train_data = Dataset.from_dict({"text": X_train, "label": y_train})
val_data = Dataset.from_dict({"text": X_val, "label": y_val})

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(b):
    return tokenizer(b["text"], padding="max_length", truncation=True, max_length=32)

train_data = train_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)
train_data.set_format("torch", columns=["input_ids","attention_mask","label"])
val_data.set_format("torch", columns=["input_ids","attention_mask","label"])

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(INTENT_LABELS))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# FIXED: use eval_strategy
training_args = TrainingArguments(
    output_dir="./intent_model_results",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",   # <--- this is the correct parameter name
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

trainer.train()

os.makedirs("models/intent_model", exist_ok=True)
model.save_pretrained("models/intent_model")
tokenizer.save_pretrained("models/intent_model")

eval_results = trainer.evaluate()
print("\n" + "="*55)
print("INTENT CLASSIFIER RESULTS")
print("="*55)
print(f"Accuracy : {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall   : {eval_results['eval_recall']:.4f}")
print(f"F1-Score : {eval_results['eval_f1']:.4f}")
print("="*55)

# Confusion matrix
preds = trainer.predict(val_data)
y_pred = np.argmax(preds.predictions, axis=-1)
cm = confusion_matrix(preds.label_ids, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=INTENT_LABELS, yticklabels=INTENT_LABELS)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")
print("\nTraining complete. Run 'python main.py' to start the API.")