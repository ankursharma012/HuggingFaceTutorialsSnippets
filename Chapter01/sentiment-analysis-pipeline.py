from transformers import pipeline

classifier = pipeline("sentiment-analysis")
results = classifier("I have been waiting for a hugging face course my whole life")

for result in results:
    label = result['label']
    score = result['score']
    print(f"The label is {label} and the score is {score*100:.4f}%")



# Using classifier with several user-input sentences

lstOfSentences = []
results2 = []
prompt = input("Enter sentences that you want to analyse sentiments for, else 'q' to quit: ")
while(prompt != 'q'):
    lstOfSentences.append(prompt)
    prompt = input()
if lstOfSentences.count != 0:
    results = classifier(lstOfSentences)

for i in range(0,len(lstOfSentences)):
    feedback = lstOfSentences[i]
    label = results[i]['label']
    score = results[i]['score']
    print(f"The sentence is '{feedback}' and label={label}, and score={score*100:.2f}%")

