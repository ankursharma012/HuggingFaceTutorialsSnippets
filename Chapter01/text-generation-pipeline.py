from transformers import pipeline

generator = pipeline("text-generation")
generated_text = generator("I am learning this course because")
print(generated_text)