# coding=utf-8
import json
import csv
import pandas as pd

label_list = ["0", "1", "-1"]
puncs = ['【', '】', ')', '(', '、', '，', '“', '”',
         '。', '《', '》', ' ', '-', '！', '？', '.',
         '\'', '[', ']', '：', '/', '.', '"', '\u3000',
         '’', '．', ',', '…', '?', ';', '·', '%', '（',
         '#', '）', '；', '>', '<', '$', ' ', ' ', '\ufeff']

train_list = []

# new_train = train_f = open("../Pseudo-Label/new.csv",'r')
# for line in train_f:
# 	a = line.split(",")
# 	train_list.append([a[0], a[1].strip()])
################
csvFile1 = open("chn_train.csv", 'w', newline='')
writer1 = csv.writer(csvFile1)
writer1.writerow(('微博id', '微博发布时间', '发布人账号', '微博中文内容', '微博图片', '微博视频', '情感倾向'))
csvFile2 = open("chn_test.csv", 'w', newline='')
writer2 = csv.writer(csvFile2)
writer2.writerow(('微博id', '微博发布时间', '发布人账号', '微博中文内容', '微博图片', '微博视频', '情感倾向'))
# train_f = open("./train.tsv", 'r')
count_0 = 0
count_1 = 0
count_11 = 0
train_label = open('./train.tsv','r')
dev_label = open('./dev.tsv','r')
test_label = open('./test.tsv','r')
y_pred = []
y_true = []
count_pred = 0
count_true = 0
for line in train_label:
    # print(len(line.strip().split(",")))
    label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    a = label[0].split('\t')
    if count_true > 0:
        writer1.writerow((count_true, '', '', a[1], '', '', int(a[0])))
    count_true += 1
for line in dev_label:
    # print(len(line.strip().split(",")))
    label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    a = label[0].split('\t')
    if count_true > 0 and count_pred>0:
        writer1.writerow((count_true, '', '', a[1], '', '', int(a[0])))
    count_pred += 1

count_test = 0
for line in test_label:
    # print(len(line.strip().split(",")))
    label = next(csv.reader(line.splitlines(),skipinitialspace=True))
    a = label[0].split('\t')
    if count_test > 0:
        writer2.writerow((count_test, '', '', a[1], '', '', int(a[0])))
    count_test += 1
