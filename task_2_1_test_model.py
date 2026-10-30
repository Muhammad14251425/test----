"""Task 2.1 – load the zero-shot pipeline and run one quick test.

Run in Google Colab:
    !pip install -q transformers torch
    !python task_2_1_test_model.py
"""

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("Model used:", classifier.model.config._name_or_path)
print("Architecture:", classifier.model.config.architectures)

result = classifier(
    "The new phone has an amazing camera but the battery dies before lunch.",
    candidate_labels=["technology", "sports", "politics"],
)

print("\nSequence:", result["sequence"])
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:<12} {score:.4f}")
