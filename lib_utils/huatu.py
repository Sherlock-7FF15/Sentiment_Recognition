import matplotlib.pyplot as plt
import numpy as np
import pylab as mpl  # import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

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
    ax.set_xticklabels(("BIGRU_CNN_Att", "BIGRU", "GRU", "CNN_Att", "CNN"))
    # plt.legend(loc='lower right', fontsize=40)  # 标签位置
    # ax.plot(ind, average_accuracy, label='$average_accuracy$')
    # ax.plot(ind, average_recall, label='$average_recall$')
    # ax.plot(ind, max_F1_score, label='$max_F1_score$')

    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()

    plt.show()


def plot_line(epoch,a2_list,a3_list,a4_list,a5_list,a6_list):
    plt.figure()
    label1 = "one_layer_BIGRU"
    label2 = "BIGRU_CNN_Att"
    label3 = "BIGRU"
    label4 = "GRU"
    label5 = "CNN_Att"
    label6 = "CNN"

    # plt.plot(range(1, epoch+2), a1_list, 'o--b', label=label1)  # 蓝色
    plt.plot(range(1, epoch+2), a2_list, '<--r', label=label2)  # 黑色
    plt.plot(range(1, epoch+2), a3_list, '+--y', label=label3)  # 黄色
    plt.plot(range(1, epoch+2), a4_list, 'x--k', label=label4)  # 红色
    plt.plot(range(1, epoch+2), a5_list, '>--g', label=label5)  # 绿色
    plt.plot(range(1, epoch+2), a6_list, '.--c', label=label6)  # 青色
    # plt.plot(range(1, len(acc) + 2), val_loss, 'b', label='Validation loss')
    plt.title('五种模型F1_score比较')
    plt.xlabel('Iteration')
    plt.ylabel('F1_score')
    plt.legend()

    plt.show()


def plot_PRC(epoch,a2_list,a3_list,a4_list,a5_list,a6_list):
    plt.figure()
    label1 = "one_layer_BIGRU"
    label2 = "BIGRU_CNN_Att"
    label3 = "BIGRU"
    label4 = "GRU"
    label5 = "CNN_Att"
    label6 = "CNN"

    # plt.plot(range(1, epoch+2), a1_list, 'o--b', label=label1)  # 蓝色
    plt.plot(range(1, epoch + 2), a2_list, '<--r', label=label2)  # 黑色
    plt.plot(range(1, epoch + 2), a3_list, '+--y', label=label3)  # 黄色
    plt.plot(range(1, epoch + 2), a4_list, 'x--k', label=label4)  # 红色
    plt.plot(range(1, epoch + 2), a5_list, '>--g', label=label5)  # 绿色
    plt.plot(range(1, epoch + 2), a6_list, '.--c', label=label6)  # 青色
    # plt.plot(range(1, len(acc) + 2), val_loss, 'b', label='Validation loss')
    plt.title('五种模型Loss比较')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# 文本长度对F1值的影响
def plot_a_line(epoch,a1_list):
    plt.figure()
    label1 = "one_layer_BIGRU"

    plt.plot(epoch, a1_list, 'o--b')  # 蓝色
    # plt.title('五种模型F1_score比较')
    x_major_locator = MultipleLocator(20)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(80, 300)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5, 110)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.xlabel('MaxLen')
    plt.ylabel('F1_score')
    # plt.legend()

    plt.show()
# 画图
average_accuracy = [0.9576, 0.9380, 0.9053, 0.9327, 0.8970]
average_recall = [0.9202, 0.9143,  0.9331, 0.9033, 0.9265]
max_F1_score = [0.9398, 0.9241, 0.9184, 0.9178, 0.9069]
plot_bar(average_accuracy, average_recall, max_F1_score)  # 平均准确率，平均召回率，F1值：柱状图
#
# a1_list = [0.8867, 0.9532, 0.9487, 0.9559, 0.9555, 0.9552, 0.9543, 0.9583, 0.9506, 0.9517, 0.9522, 0.9586, 0.9562,
#            0.9552, 0.9558]
# BIGRU_CNN_Att
a2_list = [0.8067, 0.9215, 0.9323, 0.9362, 0.9323, 0.9323, 0.9312, 0.9355, 0.9329, 0.9277, 0.9357, 0.9303, 0.9391,
           0.9358, 0.9398]
# two_layer_BIGRU
a3_list = [0.8304, 0.9127, 0.9268, 0.9299, 0.9186, 0.9292, 0.9257, 0.9256, 0.9180, 0.9087, 0.9180, 0.9280, 0.9280,
           0.9252, 0.9241]
# [0.8875, 0.7349, 0.8825, 0.8960, 0.9127, 0.9214, 0.9299, 0.9268, 0.9152, 0.9383, 0.9317, 0.9378, 0.7545,
#            0.9369, 0.9384]
# GRU
a4_list = [0.8175, 0.8952, 0.9054, 0.8553, 0.9173, 0.9224, 0.9192, 0.9197, 0.9245, 0.8972, 0.9155, 0.9233, 0.9289,
           0.9147, 0.9184]
# CNN_Att
a5_list = [0.6888, 0.8290, 0.9173, 0.9277, 0.9239, 0.9243, 0.9233, 0.9185, 0.9121, 0.9172, 0.9154, 0.9189, 0.9178,
           0.9078, 0.9178]
# CNN
a6_list = [0.6883, 0.9050, 0.9176, 0.9305, 0.9087, 0.9254, 0.9237, 0.9165, 0.9229, 0.9202, 0.9202, 0.9193, 0.9246,
           0.9194, 0.9169]
epoch = 14
print('长度',len(a2_list),len(a3_list),len(a4_list))
# 五个模型的f1值
plot_line(epoch, a2_list, a3_list, a4_list, a5_list, a6_list)  # 六个模型的F1值折线图

# BIGRU_CNN_Att
a2_loss = [0.7433, 0.4181, 0.3371, 0.3305, 0.3257, 0.3622, 0.3700, 0.3719, 0.3889, 0.3860, 0.3957, 0.4062, 0.3844,
           0.4020, 0.3909]
# two_layer_BIGRU
a3_loss = [0.6590, 0.4183, 0.3892, 0.3960, 0.3818, 0.4092, 0.4027, 0.3998, 0.4223, 0.4288, 0.4386, 0.4307, 0.4727,
           0.4496, 0.4484]
# [0.8875, 0.7349, 0.8825, 0.8960, 0.9127, 0.9214, 0.9299, 0.9268, 0.9152, 0.9383, 0.9317, 0.9378, 0.7545,
#            0.9369, 0.9384]
# GRU
a4_loss = [0.7823, 0.5045, 0.4468, 0.4540, 0.4263, 0.4388, 0.4324, 0.4182, 0.4113, 0.4058, 0.4052, 0.4424, 0.3943,
           0.4155, 0.4162]
# CNN_Att
a5_loss = [0.8976, 0.5265, 0.3769, 0.3446, 0.4133, 0.3681, 0.4669, 0.5118, 0.5628, 0.5491, 0.6300, 0.5763, 0.6652,
           0.5936, 0.6009]
# CNN
a6_loss = [0.8195, 0.4761, 0.4426, 0.4494, 0.4356, 0.4902, 0.4828, 0.5185, 0.6073, 0.6057, 0.5518, 0.5774, 0.5952,
           0.6297, 0.6340]
# 五个模型的loss值
plot_PRC(epoch, a2_loss, a3_loss, a4_loss, a5_loss, a6_loss)  # 六个模型的F1值折线图

# 文本长度对F1值的影响
a6_loss = [0.9247, 0.9269, 0.9197, 0.9276, 0.9273, 0.9398, 0.9272, 0.9274, 0.9273, 0.9271]
a_list = [80,120,140,160,180,200,220,240,260,280]
plot_a_line(a_list,a6_loss)
# plot_PRC(recall_1, precision_1, recall_2, precision_2, recall_3, precision_3,
#          recall_4, precision_4, recall_5, precision_5, recall_6, precision_6,)  # 六个模型的召回率与准确率的PRC图

