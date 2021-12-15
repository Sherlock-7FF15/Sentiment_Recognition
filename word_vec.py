from collections import Counter
import collections
import jieba
import pandas as pd
import numpy as np
import os
import utils as utl  # 导进来主要是用于标签的转码

def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    # 读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        line = fStopWords.readline()
        while line:
            stop_words.append(line[:-1]) # 去\n
            line = fStopWords.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，去除停用词，得到词典
    sFolderPath = 'JinYong\'s Works'
    lsFiles = []
    for root, dirs, files in os.walk(sFolderPath):
        for file in files:
            if file.endswith(".txt"):
                lsFiles.append(os.path.join(root, file))
    raw_word_list = []
    for item in lsFiles:
        with open(item, "r", encoding='UTF-8') as f:
            line = f.readline()
            while line:
                while '\n' in line:
                    line = line.replace('\n', '')
                while ' ' in line:
                    line = line.replace(' ', '')
                # 如果句子非空
                if len(line) > 0:
                    raw_words = list(jieba.cut(line, cut_all=False))
                    for item in raw_words:
                        # 去除停用词
                        if item not in stop_words:
                            raw_word_list.append(item)
                line = f.readline()
    return raw_word_list

words = read_data()
print('Data size', len(words))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 100000 # 100000
def build_dataset(words):
    # 词汇编码
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count", len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 使用生产的词汇编码将前面产生的 string list[words] 转变成 num list[data]
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # 反转字典 key为词汇编码 values为词汇本身
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
#删除words节省内存
del words
print('Most common words ', count[1:6])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
def initialize_inputs():
    # 处理中文数据
    # training_file = 'word.txt'
    # training_data = get_ch_label(training_file)
    # # print(training_data[:1000])
    # training_ci = fenci(training_data)
    # # print(training_ci)
    # print("总字数", len(training_data))
    # print("总词数", len(training_ci))
    # # vocab_to_int, int_to_vocab = utl.create_lookup_tables(training_ci)
    # #
    # # print('字典相互对应',len(vocab_to_int),len(int_to_vocab),vocab_to_int['母'])
    # # self.vocab_size = len(vocab_to_int) + 1
    # # print('字典大小',self.vocab_size)
    # training_label, count, dictionary, words = build_dataset(training_ci, 50920)
    words = read_data()
    data, count, dictionary, reverse_dictionary = build_dataset(words)

    words_size = len(dictionary)
    vocab_size = len(dictionary) + 1
    print('.....', dictionary)
    print("字典词数", words_size)
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # for i in training_ci[:10]:
    #     print(i)
    training_file = 'word.txt'
    messages = get_ch_label_two(training_file)
    a = []
    for message in messages:
        a.append(line_qu_ting(message.strip()))
    for i in a[:10]:
        print('......',i)
    messages = utl.encode_ST_messages(a, dictionary)
    messages_len = Counter([len(x) for x in messages])
    print('Zero-length messages: {}'.format(messages_len[0]))
    print("Maximum message length: {}".format(max(messages_len)))
    ######
    neg = pd.read_excel('data/neg.xls', header=None, index=None)
    pos = pd.read_excel('data/pos.xls', header=None, index=None)
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg) + 19)), axis=0)
    print('------', len(messages), len(labels))
    messages, labels = utl.drop_empty_messages(messages, labels)

    # Pad Messages，把数据都对齐，全部变为长度为244
    messages = np.array(messages)
    print(messages.shape)
    messages = utl.zero_pad_messages(messages, seq_len=400)
    # for i in messages[:2]:
    #     print(i)

    # Train,Test,Validation Split
    # 从这里控制可训练的数据规模

    return labels,messages

# 分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)  # 默认是精确模式
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    # 以空格将字符串分开
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci

def get_ch_label(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('utf-8')

    return labels

def get_ch_label_two(txt_file):

    labels = []
    with open(txt_file, 'rb') as f:
        for label in f:
            labels.append(label.decode('utf-8'))

    return labels

def line_qu_ting(message):
    # 读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        line = fStopWords.readline()
        while line:
            stop_words.append(line[:-1])  # 去\n
            line = fStopWords.readline()
    stop_words = set(stop_words)
    # print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    line = message
    raw_word_list = []
    raw_words = list(jieba.cut(line, cut_all=False))
    for item in raw_words:
        # 去除停用词
        if item not in stop_words:
            raw_word_list.append(item)
    return raw_word_list

# def build_dataset(words, n_words):
#     """Process raw inputs into a dataset."""
#     count = [['UNK', -1]]
#     # print(collections.Counter(words))
#     # Counter({'，': 89, '的': 78, '。': 55, '就': 19, '电池': 14, '会': 13, '运动': 13, '是': 11, '身体': 11, '可以': 11,})
#     count.extend(collections.Counter(words).most_common(n_words - 1))
#     dictionary = dict()
#     for word, _ in count:
#         dictionary[word] = len(dictionary)
#     data = list()
#     unk_count = 0
#     for word in words:
#         if word in dictionary:
#             index = dictionary[word]
#         else:
#             index = 0
#             unk_count += 1
#         data.append(index)
#     count[0][1] = unk_count
#
#     reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#     return data, count, dictionary, reversed_dictionary

if __name__ == '__main__':
    labels, messages = initialize_inputs()
    print(labels,messages)