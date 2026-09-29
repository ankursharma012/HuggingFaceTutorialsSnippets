from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I am enjoying learning AI and working on something good.",
    "I literally hate HuggingFace library!"
]

# Preprocessing steps
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)


# Model download
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)


#post-processing
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)