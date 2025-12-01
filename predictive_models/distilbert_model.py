# predictive_models/distilbert_model.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict

print("\n========= GPU STATUS =========")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU being used:", torch.cuda.get_device_name(0))
print("================================\n")
# 1. Load dataset

DATA_PATH = "project_data/dbert_ready_data.csv"

df = pd.read_csv(DATA_PATH)

print("Loaded dataset:")
print(df.head())
print("Class distribution:")
print(df["class"].value_counts())


# 2. Encode labels (NEG / NEU / POS -> 0 / 1 / 2)


label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["class"])


# 3. Train / Val / Test split (same as your other model)


X = df["text"].tolist()
y = df["label"].tolist()

# first split: 80% temp train, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# second split: from temp split out validation (25% of temp = 20% overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print("Split sizes:")
print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

# Convert into HuggingFace datasets
train_df = pd.DataFrame({"text": X_train, "label": y_train})
val_df = pd.DataFrame({"text": X_val, "label": y_val})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# 4. Tokenizer


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")


# 5. Load model


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)
print("Using device:", model.device)

# 6. Training Arguments


training_args = TrainingArguments(
    output_dir="../project_data/distilbert_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    num_train_epochs=1,
    load_best_model_at_end=True,                
    dataloader_pin_memory=True,
    dataloader_num_workers=0,       

)


# 7. Trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

trainer = Trainer(
    model=model.to("cuda"),
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# 8. Train


trainer.train()

# Save final model
trainer.save_model("../project_data/distilbert_model")


# 9. Evaluate on Test Set


pred_output = trainer.predict(dataset["test"])
pred_labels = np.argmax(pred_output.predictions, axis=1)

acc = accuracy_score(y_test, pred_labels)
f1 = f1_score(y_test, pred_labels, average="weighted")

print("\n=========== FINAL TEST RESULTS ===========")
print("Test Accuracy:", acc)
print("Test F1 Score:", f1)
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, target_names=label_encoder.classes_))

