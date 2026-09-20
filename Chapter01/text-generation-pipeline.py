from transformers import pipeline

print("Loading SmolLM2-360M text generation pipeline...")
# Explicitly specify the lightweight SmolLM2 model
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")

print("Generating text...")
generated_text = generator(
    "I am learning this course because", 
    max_new_tokens=30,
    num_return_sequences=1
)

print("Done!")
print(generated_text)