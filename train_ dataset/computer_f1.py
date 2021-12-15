from sklearn.metrics import f1_score
import csv
true_label = open('./10k.csv','r')
pred_label = open('./pytorchresult.csv','r')
y_pred = []
y_true = []
count_pred = 0
count_true = 0
for line in true_label:
    # print(len(line.strip().split(",")))
    id, trigger, object, subject, time, location,label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    if count_true > 0:
        y_true.append(int(label))
    count_true += 1

for line in pred_label:
    # print(len(line.strip().split(",")))
    id, label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    if count_pred > 0:
        y_pred.append(int(label))
    count_pred += 1

print(f1_score(y_true, y_pred, average='macro'))
# print(f1_score(y_true, y_pred, average='weighted'))
