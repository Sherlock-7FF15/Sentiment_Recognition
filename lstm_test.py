#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
import jieba
import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

import yaml
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 200

from lib_utils import model_GAN_CNN_BIGRU_Att
from lstm_train import loadfile,tokenizer,word2vec_train,get_data



def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

checkpoint_dir = "model/"
def lstm_predict(string1,string2,string3):
    print('loading model......')
    # 训练模型，并保存
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print("x_train.shape and y_train.shape:")
    print(x_train.shape, y_train.shape, n_symbols)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        model = model_GAN_CNN_BIGRU_Att.LstmTFModel(useAttention=True, restore=False,
                                                    index_dict=index_dict, word_vectors=word_vectors, combined=combined,
                                                    y=y, embedding_weights=embedding_weights)
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.Saver()

        if ckpt != None:
            saver.restore(sess, ckpt)
            print("Reading model parameters from {0}".format(ckpt))
        model.batchSize = 3
        print('-------',model.batchSize)
        data1=input_transform(string1)
        data2=input_transform(string2)
        data3=input_transform(string3)

        data1.reshape(1,-1)
        data2.reshape(1,-1)
        data3.reshape(1,-1)

        print(np.array(data1))
        print(data2)
        data = np.concatenate((data1,data2,data3),axis=0)
        prob = False
        print(data)
        model.initialize_model(3)

        result = model.predict(sess, data)
        print('----------',result)
        # result=model.predict_classes(data)
        # print result # [[1]]
        if result[0]==1:
            print(string1,' positive')
        elif result[0]==0:
            print(string1,' neural')
        else:
            print(string1,' negative')


if __name__=='__main__':

    string2='酒店的环境非常好，价格也便宜，值得推荐'
    string3='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string = "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好"
    # string = "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！"
    string1 = "书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。"
    # string2 = "不是太好"
    # string = "不错不错"
    # string = "真的一般，没什么可以学习的"
    # string = '质量一般'

    lstm_predict(string1,string2,string3)