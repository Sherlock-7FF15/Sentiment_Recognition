#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import tensorflow as tf
import pandas as pd 
import numpy as np 
import jieba
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
import keras
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import collections

from matplotlib.font_manager import FontProperties
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml

# set parameters:
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 300
n_iterations = 5  # ideally more..
n_exposures = 10  # 所有频数超过10的词语
window_size = 5
# n_epoch = 4
# input_length = 100
maxlen = 200

batch_size = 32


# 第一步 # combined,y=loadfile()
def loadfile():
    # neg=pd.read_csv('data/neg.csv',header=None,index_col=None)
    # pos=pd.read_csv('data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    # neu=pd.read_csv('../data/neutral.csv', header=None, index_col=None)

    content = pd.read_csv('train_ dataset/90k.csv',header=None,index_col=None)
    a = []
    for k, v in enumerate(content[3][1:]):
        # 添加清洗功能，清洗完了再添加
        if content[6][k + 1] == '-1' or content[6][k + 1] == '0' or content[6][k + 1] == '1':
            a.append(v)
    combined = np.array(a)
    # print('bbbbb', combined)
    # 处理标签
    b = []
    for j in content[6][1:]:
        if j == '-1' or j == '0' or j == '1':
            b.append(int(j))
    y = np.array(b)
    return combined, y

    # combined = np.concatenate((pos[0], neg[0]))
    # y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)


#  第二步  对句子进行分词，并去掉换行符 # combined = tokenizer(combined)
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    # 读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        line = fStopWords.readline()
        while line:
            stop_words.append(line[:-1])  # 去\n
            line = fStopWords.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
    # text = [jieba.lcut(document.replace('\n', '')) for document in text]
    raw_word_list = []
    for document in text:
        # line = f.readline()
        # 如果句子非空
        if len(document) > 0:
            a = []
            # 用空格替代换行符，raw_words为二维
            raw_words = list(jieba.cut(document.replace('\n', '')))
            for item in raw_words:
                # 去除停用词
                if item not in stop_words:
                    a.append(item)
        raw_word_list.append(a)
    return raw_word_list


def create_dictionaries(model=None, combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        print("gensim_dict: ",list(gensim_dict))
        # print('**',model.wv.vocab.keys())
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        #  freqxiao10->0 所以k+1
        # 所有频数超过10的词语的索引,(k->v)=>(v->k)
        print('--',gensim_dict.items())
        # 将k和v反转  0为所有频数小于10的词，所以索引从1开始
        # 词：索引+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        # print("w2indx: ",w2indx)
        # 所有频数超过10的词语的词向量, (word->model(word))
        # 词：词向量
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined): # 闭包-->临时使用
            '''
            Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freqxiao10->0
                data.append(new_txt)
            # 得到每条微博中每个词的索引 [[],[],[]]
            # print("data: ",data)
            return data  # word=>index
        combined = parse_dataset(combined)
        # 从这个位置，文本内容就变成句子中词语的索引表示
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        # 文本长短不一，统一长度为200,
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# 第三步 # index_dict, word_vectors,combined=word2vec_train(combined)
#  创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,  #300  词的维度
                     min_count=n_exposures,  # 训练所有频数超过10的词语
                     window=window_size,  # 7
                     sample=1e-5,  #负采样
                     workers=cpu_count,
                     iter=n_iterations)  # 训练次数5  效果不好可增大
    model.build_vocab(combined)  # input: list
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
    model.save('lstm_data_test/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


# 第四步
# n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict,
# word_vectors,combined,y)
def get_data(index_dict, word_vectors, combined, y):
    # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    n_symbols = len(index_dict) + 1
    # 初始化 索引为0的词语，词向量全为0（8306，200)
    print('n_symbols',n_symbols)
    embedding_weights = np.zeros((n_symbols, vocab_dim), dtype=np.float32)
    # 从索引为1的词语开始，对每个词语对应其词向量
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    # 得到24596个词语的200维词向量
    # print("embedding_weights: ",embedding_weights)
    # 训练集和测试集分为8:2，二分类，可不要，在keras中会重新定义
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=2)
    y_test = keras.utils.to_categorical(y_test,num_classes=2)
    # print x_train.shape,y_train.shape
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='image/tsne3.png', fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     fontproperties=fonts,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename, dpi=800)


plot_histograms = True
def analysisfile(source_file):
    # 分析文本
    # print('source',source)
    num_words = []
    for i in source_file:
        counter = len(i)
        num_words.append(counter)
    print('num_word: ',num_words)  #每条文本的长度
    # if counter % 100000 == 0:
    #     print("  reading data line %d" % counter)
    #     sys.stdout.flush()
    nue = collections.Counter(num_words)  #同样的文本长度出现的次数
    print('..',nue[0])
    if plot_histograms:
        plot_histo_lengths("source_lengths", num_words)


def plot_histo_lengths(title, lengths):
    sigma = np.std(lengths)  # 标准差
    mu = np.mean(lengths)  # 平均值
    x = np.array(lengths)
    # n, bins, patches = plt.hist(x, 25, facecolor='g', alpha=0.5)
    # y = mlab.normpdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线
    # plt.plot(bins, y, 'r--')
    # plt.title(title)
    plt.xlabel("文本长度")
    plt.ylabel("频次")
    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
    # Create a histogram by providing the bin edges (unequally spaced).
    bins = [0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75, 82.5,
            90, 97.5, 105, 112.5, 120, 127.5, 135, 142.5, 150, 157.5,
            165, 172.5, 180, 187.5, 195, 202.5, 210, 217.5, 225,
            232.5, 240, 247.5, 255, 262.5, 270, 277.5, 285, 292.5, 300]
    plt.hist(x, bins, histtype='bar', rwidth=0.8)
    plt.ylim([0, 20000])
    plt.show()

# ##定义网络结构
# def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
#     print('Defining a Simple Keras Model...')  # input_length=100
#     model = Sequential()  # or Graph or whatever
#     model.add(Embedding(output_dim=vocab_dim,  # 100
#                         input_dim=n_symbols,  # 8305
#                         mask_zero=True,
#                         weights=[embedding_weights],
#                         input_length=input_length))  # Adding Input Length
#     embe = Embedding(output_dim=vocab_dim,
#                         input_dim=n_symbols,
#                         mask_zero=True,
#                         weights=[embedding_weights],
#                         input_length=input_length).input_dim
#     print('embe',embe)
#     model.add(LSTM(output_dim=50, activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(Dense(3, activation='softmax')) # Dense=>全连接层,输出维度=3
#     model.add(Activation('softmax'))
#
#     print('Compiling the Model...')
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',metrics=['accuracy'])
#
#     print("Train...") # batch_size=32
#     model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)
#
#     print("Evaluate...")
#     score = model.evaluate(x_test, y_test,
#                                 batch_size=batch_size)
#
#     yaml_string = model.to_yaml()
#     with open('../model/lstm.yml', 'w') as outfile:
#         outfile.write(yaml.dump(yaml_string, default_flow_style=True) )
#     model.save_weights('../model/lstm.h5')
#     print('Test score:', score)
#
#
#训练模型，并保存
# print('Loading Data...')
# combined,y=loadfile()
# print(len(combined),len(y))
# print('Tokenising...')
# combined = tokenizer(combined)
# print('Training a Word2vec model...')
# index_dict, word_vectors,combined=word2vec_train(combined)
#
# print('Setting up Arrays for Keras Embedding Layer...')
# n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
# print("x_train.shape and y_train.shape:")
# print(x_train.shape,y_train.shape,n_symbols)
# # train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)