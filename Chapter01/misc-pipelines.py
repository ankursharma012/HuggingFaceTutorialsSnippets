from transformers import pipeline

# Mask Filling
unmasker = pipeline("fill-mask")
unmasked_sentence = unmasker("This course will teach you all about <mask> models", top_k=2)

outputs = []
for unmasked_sen in unmasked_sentence:
    outputs.append(unmasked_sen['sequence'])

print(outputs)

# Named Entity Recognition
ner = pipeline("ner", model="dslim/distilbert-NER")
nerd = ner("I am Ankur born in Rewari and worked at General Electric Helathcare Qatar")
print(nerd)

#Question answering - Answer from the provided context - doesn't work on latest pipelines
question_answerer = pipeline("question-answering", model="distilbert/distilbert-base-uncased-distilled-squad")
print(question_answerer(
    question="Where do I live ?",
    context="My name is Ankur and I work at GE In Qatar"
    ))




