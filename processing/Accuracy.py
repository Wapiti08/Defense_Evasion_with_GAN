from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

'''
accuracy=accuracy_score(labels,guesses)
precision=precision_score(labels,guesses)
recall=recall_score(labels,guesses)
f_1=f1_score(labels,guesses)

print(accuracy)
print(precision)
print(recall)
print(f_1)
'''

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
  #True Positives
  if labels[i] == 1 and guesses[i] == 1:
    true_positives += 1
  #True Negatives
  if labels[i] == 0 and guesses[i] == 0:
    true_negatives += 1
  #False Positives
  if labels[i] == 0 and guesses[i] == 1:
    false_positives += 1
  #False Negatives
  if labels[i] == 1 and guesses[i] == 0:
    false_negatives += 1
    
accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)

recall = true_positives / (true_positives + false_negatives)
print(recall)

precision = true_positives / (true_positives + false_positives)

#the weights of P and R are the same 
f_1 = 2*precision*recall/(precision+recall)
print(precision)
