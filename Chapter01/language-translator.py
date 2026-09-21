from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_id = "Helsinki-NLP/opus-mt-en-fr"

print("Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = "Hello, how are you learning AI Engineering today"

print("Tokenizing input text..")
inputs = tokenizer(text, return_tensors="pt", truncation=True)

print("Translating")
with torch.no_grad():
    translated_tokens = model.generate(**inputs, max_new_tokens=128)

translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("Done!")
print(f"Translated text: {translated_text}")