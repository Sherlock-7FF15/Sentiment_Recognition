import matplotlib.pyplot as plt
import numpy as np
import pylab as mpl  # import matplotlib as mpl

# 设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_bar(average_accuracy, average_recall, max_F1_score):
    print('-', average_accuracy)
    print('--', average_recall)
    print('---', max_F1_score)
    ind = np.arange(len(average_accuracy))  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind + width*ind, average_accuracy, width, label='average_accuracy')
    rects2 = ax.bar(ind + width*(ind+1), average_recall, width, label='average_recall')
    rects3 = ax.bar(ind + width*(ind+2), max_F1_score, width, label='max_F1_score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by ChnSentiCorp and Crawled Dataset Samples')
    ax.set_ylim(.0, 1.)

    index = [0.25, 1.5, 2.75, 3.975, 5.225, 6.5]
    ax.set_xticks(index)
    ax.set_xticklabels(("one_layer_BIGRU", "BIGRU_CNN_Att", "GRU",
                        "two_layer_BIGRU", "CNN_Att", "CNN"))
    # plt.legend(loc='lower right', fontsize=40)  # 标签位置
    # ax.plot(ind, average_accuracy, label='$average_accuracy$')
    # ax.plot(ind, average_recall, label='$average_recall$')
    # ax.plot(ind, max_F1_score, label='$max_F1_score$')

    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()

    plt.show()


def plot_line(epoch,a1_list,a2_list,a3_list,a4_list,a5_list,a6_list):
    plt.figure()
    label1 = "one_layer_BIGRU"
    label2 = "BIGRU_CNN_Att"
    label3 = "GRU"
    label4 = "two_layer_BIGRU"
    label5 = "CNN_Att"
    label6 = "CNN"

    plt.plot(range(1, epoch+2), a1_list, 'o--b', label=label1)  # 蓝色
    plt.plot(range(1, epoch+2), a2_list, 'x--k', label=label2)  # 黑色
    plt.plot(range(1, epoch+2), a3_list, '+--y', label=label3)  # 黄色
    plt.plot(range(1, epoch+2), a4_list, '<--r', label=label4)  # 红色
    plt.plot(range(1, epoch+2), a5_list, '>--g', label=label5)  # 绿色
    plt.plot(range(1, epoch+2), a6_list, '.--c', label=label6)  # 青色
    # plt.plot(range(1, len(acc) + 2), val_loss, 'b', label='Validation loss')
    plt.title('六种模型F1_score比较')
    plt.xlabel('Iteration')
    plt.ylabel('F1_score')
    plt.legend()

    plt.show()


def plot_PRC(recall_1,precision_1,recall_2, precision_2,recall_3, precision_3,recall_4, precision_4,recall_5, precision_5,
             recall_6, precision_6):
    plt.figure()

    plt.plot(recall_1, precision_1, 'o--b', label='one_layer_BIGRU')
    plt.plot(recall_2, precision_2, 'x--k', label='BIGRU_CNN_Att')
    plt.plot(recall_3, precision_3, '+--y', label='GRU')
    plt.plot(recall_4, precision_4, '<--r', label='two_layer_BIGRU')
    plt.plot(recall_5, precision_5, '>--g', label='CNN_Att')
    plt.plot(recall_6, precision_6, '.--c', label='CNN')

    # plt.plot(range(1, len(acc) + 2), val_loss, 'b', label='Validation loss')
    plt.title(u'PRC曲线')
    plt.xlabel('召回率')
    plt.ylabel('准确率')
    plt.legend()
    plt.show()


# average_accuracy = []
# average_recall = []
# max_F1_score = []
# a1_list = []
# a2_list = []
# a3_list = []
# a4_list = []
# a5_list = []
# a6_list = []
# #######################
# recall_1 = []
# precision_1 = []
# #
# recall_2 = []
# precision_2 = []
# #
# recall_3 = []
# precision_3 = []
# #
# recall_4 = []
# precision_4 = []
# #
# recall_5 = []
# precision_5 = []
# #
# recall_6 = []
# precision_6 = []
# # 平均准确率，平均召回率，F1值：柱状图
# plot_bar(average_accuracy, average_recall, max_F1_score)
# # 六个模型的F1值折线图
# plot_line(100, a1_list, a2_list, a3_list, a4_list, a5_list, a6_list)
# # 六个模型的召回率与准确率的PRC图
# plot_PRC(recall_1, precision_1, recall_2, precision_2, recall_3, precision_3,
#          recall_4, precision_4, recall_5, precision_5, recall_6, precision_6,)