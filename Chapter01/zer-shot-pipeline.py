from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
                    "This is a course about the transformer library",
                    candidate_labels=["education", "tutorial", "polictics"],
                    )
print(result)