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
csvFile = open("90k.csv", 'w', newline='')
writer = csv.writer(csvFile)
writer.writerow(('微博id', '微博发布时间', '发布人账号', '微博中文内容', '微博图片', '微博视频', '情感倾向'))

train_f = open("../dataset/nCoV_100k_train.labled.csv", 'r')
count_0 = 0
count_1 = 0
count_11 = 0
# for line in train_f:
#     # print(len(line.strip().split(",")))
#     weiboId, weiboTime, userid, text, img, video, label = next(csv.reader(line.splitlines(),
#                                                                           skipinitialspace=True))
#     # if len(line_cols) != 7:
#     #     print(line_cols)
#     if label not in label_list:
#         continue
#     if label == '0' and count_0<5738:
#         count_0 += 1
#         writer.writerow((weiboId, weiboTime, userid, text, img, video, label))
#     if label == '1' and count_1<2560:
#         count_1 += 1
#         writer.writerow((weiboId, weiboTime, userid, text, img, video, label))
#     if label == '-1' and count_11<1706:
#         count_11 += 1
#         writer.writerow((weiboId, weiboTime, userid, text, img, video, label))
# train_list.append([text, label])
# weiboId, weiboTime, userid, text, img, video, label =  line.strip().split(",")
test_df = pd.read_csv('./10k.csv')
print('shape',test_df.shape)
tt = []
for i in test_df['微博id']:
    # print('ii',i)
    i = str(i)
    tt.append(i.strip())
for line in train_f:
    # print(len(line.strip().split(",")))
    weiboId, weiboTime, userid, text, img, video, label = next(csv.reader(line.splitlines(),
                                                                          skipinitialspace=True))
    if label not in label_list:
        continue
    if weiboId.strip() not in tt:
        writer.writerow((weiboId, weiboTime, userid, text, img, video, label))
trr_df = pd.read_csv('./90k.csv')
print('shape',trr_df.shape)
# train_end = int(len(train_list) * 0.8)
# dev_end = int(len(train_list) * 0.9)
#
# f_w_train = open("../dataset/train.csv", "w")
# for i in range(0, train_end):
#     f_w_train.write(train_list[i][0] + "\t" + train_list[i][1] + "\n")
# f_w_train.close()
#
# f_w_train = open("../dataset/dev.csv", "w")
# for i in range(train_end + 1, dev_end):
#     f_w_train.write(train_list[i][0] + "\t" + train_list[i][1] + "\n")
# f_w_train.close()

# test_f = open("../dataset/nCov_10k_test.csv",'r')
# test_list = []
# for line in test_f:
# 	# print(len(line.strip().split(",")))
# 	weiboId, weiboTime, userid, text, img, video = next(csv.reader(line.splitlines(),
# 																   skipinitialspace=True))
# 	# if len(line_cols) != 7:
# 	#     print(line_cols)
# 	test_list.append([text, "0"])
# 	# weiboId, weiboTime, userid, text, img, video, label =  line.strip().split(",")
# test_f.close()
#
# f_w_train = open("../dataset/test.csv","w")
# for i in range(len(test_list)):
# 	f_w_train.write(test_list[i][0] + "\t" + test_list[i][1] + "\n")
# f_w_train.close()
