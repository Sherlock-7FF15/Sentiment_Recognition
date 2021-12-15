# coding=utf-8
import csv
from sklearn.metrics import f1_score,accuracy_score

result_path = "./bert_wwm_ext.tsv"
out_path = "./bert_wwm_ext.csv"

# test_f = open("../dataset/rest_1.csv", 'r')
# test_list = []
# for line in test_f:
#     # print(len(line.strip().split(",")))
#     weiboId, weiboTime, userid, text, img, video,la = next(csv.reader(line.splitlines(), skipinitialspace=True))
#     # if len(line_cols) != 7:
#     #     print(line_cols)
#     test_list.append(weiboId.strip())
# # weiboId, weiboTime, userid, text, img, video, label =  line.strip().split(",")
# test_f.close()

f = open(result_path, 'r')
fw = open(out_path, "w")
cnt = 1
for line in f:
    ps = line.strip().split("\t")
    assert len(ps) == 2
    ps[0] = float(ps[0])
    ps[1] = float(ps[1])
    max_index = ps.index(max(ps))
    if max_index == 0:
        label = 0
    elif max_index == 1:
        label = 1
    fw.write(str(label) + "\n")
    cnt += 1
fw.close()


############################################
# true_label = open('./test.csv','r')
# pred_label = open('./test_results_out.csv','r')
# y_pred = []
# y_true = []
# count_pred = 0
# count_true = 0
# for line in true_label:
#     # print(len(line.strip().split(",")))
#     label,text = next(csv.reader(line.splitlines(),skipinitialspace=True))
#     if count_true > 0:
#         y_true.append(int(label))
#     count_true += 1
#
# for line in pred_label:
#     # print(len(line.strip().split(",")))
#     label = next(csv.reader(line.splitlines(),skipinitialspace=True))
#     # print('label',label)
#     y_pred.append(int(label[0]))
#
# print(accuracy_score(y_true, y_pred))