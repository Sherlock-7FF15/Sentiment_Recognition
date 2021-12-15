from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import csv

true_label = open('./test.tsv','r')
pred_label = open('./bert_wwm_ext.csv','r')
y_pred = []
y_true = []
count_pred = 0
count_true = 0
for line in true_label:
    # print(len(line.strip().split(",")))
    label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    a = label[0].split('\t')
    if count_true > 0:
        y_true.append(int(a[0]))
    count_true += 1

for line in pred_label:
    # print(len(line.strip().split(",")))
    label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    # print('label',label)
    y_pred.append(int(label[0]))

print('F1值',f1_score(y_true, y_pred,average='macro'))
print('准确率',precision_score(y_true, y_pred))
print('召回率',recall_score(y_true, y_pred))
print('正确率',accuracy_score(y_true, y_pred))