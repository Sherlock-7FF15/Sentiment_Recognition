import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import collections

from collections import Counter
import jieba
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# from lib_utils.chinese_sentiment_analysis import get_data,load_file_and_preprocessing

from lib_utils.batch_generator import BatchGenerator


# from lib_utils.attention import attention as att
# import features_word2vec

# word2vecmodel, embedding_weights, word_features, data = data_prep()
# To do tomorrow:
# 1. rename params for the alpha part
# 2. find out


class LstmTFModel:

    # initiate everything
    # embedding weights: map index to word.txt vecs
    # word_features: map
    # 每个词语的索引，词向量，以及每个句子所对应的词语索引
    def __init__(self, useAttention=True, restore=False,index_dict=None,
                 word_vectors=None,combined=None,y=None,embedding_weights=None):
        self.index_dict = index_dict
        self.word_vectors = word_vectors
        self.combined = combined
        self.y = y
        self.batchSize = 128
        self.embedding_weights = embedding_weights
        # 是否使用注意力
        self.useAttention = useAttention
        tf.reset_default_graph()

        self.session = tf.Session()
        # 是否重新加载
        self.restore = restore

        self.initialize_params()
        self.initialize_filepaths()
        # self.initialize_inputs()
        self.initialize_train_test()
        self.initialize_model()
        # self.initialize_tboard()

    # initialize all hyperparameters
    # including batch size, network size etc.
    def initialize_params(self):
        #
        self.initial_learning_rate = 0.001
        self.numClasses = 3  # 分为两类
        # self.batchSize = 64
        self.batchSize = 128
        self.maxSeqLength = 200  #
        self.embedding_size = 300
        ################################
        # 卷积使用的参数
        self.num_filters = 128  # 卷积过滤器个数
        # 卷积之后加入attention
        #################
        self.b_stddev = 0.0468
        self.h_stddev = 0.0468
        # 卷积之后加入三层全连接
        self.n_hidden_1 = 400  # 300的效果最好
        self.n_hidden_2 = 400
        self.n_hidden_3 = 400
        # self.n_hidden_5 = 784
        # 丢弃率
        self.keep_dropout = 0.98
        self.keep_prob = 0.5
        # 截断激活函数的值，防止梯度爆炸
        self.relu_clip = 20
        #####################################################
        self.num_layers = 2  # RNN层数
        self.n_hidden_units = 128  # 堆叠多层双向RNN使用
        # MBIRNN后加入attention
        self.attentionSize = 256
        # MBIRNN+attention后连接一个全局均值池化
        self.ksize = 8  # 均值池化的个数
        self.strides = 8  # 均值池化的个数
        # 正则化参数设置
        self.is_training = True
        self.grad_clip = 5.0
        self.l2_reg_lambda = 0.0001
        # 绘图设置
        self.history = {'acc': [],
                        'val_acc': [],
                        'loss': [],
                        'val_loss': [],
                        'f1_score': [],
                        'precision': [],
                        'recall': []
                        }
        # GAN参数
        self.con_dim = 2  # total continuous factor
        self.rand_dim = 20  #
        self.aelearning_rate = 0.01

    def initialize_filepaths(self):
        self.word2vecmodel_path = "./model/300features_40minwords_10context"
        self.embedding_path = "./model/embedding_weights.pkl"
        self.text2indices_path = "./model/imdb_indices.pickle"
        self.lstm_model_path = "./model/pretrained_lstm_tf.model"
        self.attention_map_path = "./figures/attention_map.png"



    # Split train and test
    def initialize_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.combined, self.y, test_size=0.2)
        # self.y_train = keras.utils.to_categorical(self.y_train,num_classes=2)
        # self.y_test = keras.utils.to_categorical(self.y_test,num_classes=2)
        # 批次生成器
        self.myBatchGenerator = BatchGenerator(self.x_train, self.y_train, self.x_test, self.y_test, self.batchSize)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.messages, self.labels,
        #                                                                         test_size=0.2)
        # print(np.array(self.X_train).shape, np.array(self.X_test).shape, np.array(self.y_train).shape,
        #       np.array(self.y_test).shape)
        # # self.X_train, self.y_train, self.X_test, self.y_test = data_split.train_test_split_shuffle(self.y, self.X,
        # # self.train_vecs, self.y_train, self.test_vecs, self.y_test                                                                                           test_size=0.1)
        # self.myBatchGenerator = BatchGenerator(self.X_train, self.y_train, self.X_test, self.y_test, self.batchSize)

    # http://aclweb.org/anthology/E17-2091
    # https://arxiv.org/pdf/1409.0473.pdf
    #
    def addAttentionToModel(self, hidden_layer):
        # hidden_layer (128,1,256)
        # [128,256]
        self.w_att = tf.Variable(tf.random_normal([128, self.attentionSize], stddev=0.1), name="w_att")
        # [256]
        self.b_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), name="b_att")
        # [256]
        self.u_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), name="u_att")

        v_att = tf.tanh(tf.matmul(tf.reshape(hidden_layer, [-1, 128]), self.w_att) \
                        + tf.reshape(self.b_att, [1, -1]))  # [256,256]
        # [256,1]
        betas = tf.nn.softmax(tf.matmul(v_att, tf.reshape(self.u_att, [-1, 1])))

        exp_betas = tf.reshape(betas, [-1, self.attentionSize])  # [1,256]
        # [1,256]
        alphas = exp_betas / tf.reshape(tf.reduce_sum(exp_betas, 1), [-1, 1])

        output = tf.reduce_sum(hidden_layer * tf.reshape(alphas,
                                                         [-1, self.attentionSize, 1]), 1)
        return output

    def initialize_model(self):
        # self.input_data = tf.placeholder(tf.float32, [128, self.maxSeqLength], name="input_data")
        # [128,200]
        self.cnn_input_data = tf.placeholder(tf.int32, [128, self.maxSeqLength], name="cnn_input_data")
        # [128,3]
        self.labels = tf.placeholder(tf.float32, [128, 3], name="labels")
        # self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.embedding = tf.Variable(self.embedding_weights,
                                     trainable=True, name='W')
        # self.input_x shape: (batch_size, sequence_length)
        # (128,200,300)  128条句子，每条句子200个词，每个词300维向量
        self.embed = tf.nn.embedding_lookup(self.embedding, self.cnn_input_data)  # (128, 256, 256)
        print('embed0：', self.embed.get_shape())  # embed0 (128, 256, 256)
        # 作用域
        with tf.name_scope('dropout'):
            self.embedding_chars_dropout = tf.nn.dropout(self.embed, self.keep_prob)
        # 定义堆叠多层双向RNN
        # Define Forward RNN Cell
        # 双向rnn
        with tf.name_scope('lstm'):
            stacked_rnn = []
            stacked_bw_rnn = []
            # num_layers=2  n_hidden_units=128
            # 创建128个双向双层GRU神经元
            for i in range(self.num_layers):
                stacked_rnn.append(tf.contrib.rnn.GRUCell(self.n_hidden_units))
                stacked_bw_rnn.append(tf.contrib.rnn.GRUCell(self.n_hidden_units))
            # 将神经元放入RNN中
            mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
            mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn)
            # layer_rnn = tf.reshape(self.embedding_chars_dropout, [self.batchSize, -1, self.n_hidden_1])
            print('layer_rnn的形状', self.embedding_chars_dropout.shape)
            outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn([mcell], [mcell_bw], self.embedding_chars_dropout, dtype=tf.float32)
            print('堆叠多层RNN的形状', outputs.shape)  # （128,200,256）
            #######################################
            # 将输入转成时间序列优先
            outputs = tf.transpose(outputs, [1, 0, 2])
            print('时间序列优先', outputs[0].shape, outputs.shape)  # （128,256,200）
            self.output = outputs[-1]
            # self.output = tf.reshape(self.output, [128, 1, 256])
            print('RNN输出的形状', self.output.shape)  # (128,256)
            self.output = tf.reshape(self.output, [self.batchSize, 1, 256])  # (128,2,256)
        # Add dropout
        with tf.name_scope('dropout'):
            # pooled_concat_flat shape: (batch_size, num_filters * len(filter_sizes))
            self.final_output = tf.contrib.layers.dropout(self.output, keep_prob=self.keep_prob,
                                                          is_training=self.is_training)
            print('卷积dropout后输出', self.final_output)  #[128,384]

            layer_3 = tf.reshape(self.final_output,[128,1,256])
            # 注意力接收三个参数
            print('没变形的layer_3', layer_3.shape)
        # 定义堆叠多层双向RNN
        # Define Forward RNN Cell
        # 加入attention，注意力机制
        with tf.variable_scope('attention'):
            cnn_attention_output = self.addAttentionToModel(layer_3)  # self.output=(128,128)
            print('attention输出的形状', cnn_attention_output.shape)  # (128,256)

        with tf.name_scope('dropout'):
            # pooled_concat_flat shape: (batch_size, num_filters * len(filter_sizes))
            cnn_attention_output = tf.contrib.layers.dropout(cnn_attention_output, keep_prob=self.keep_prob,
                                                         is_training=self.is_training)
        with tf.name_scope('fc4'):
            b4 = self.variable_on_cpu('b4', [192], tf.random_normal_initializer(stddev=self.b_stddev))
            h4 = self.variable_on_cpu('h4', [256, 192],
                                      tf.random_normal_initializer(stddev=self.h_stddev))
            layer_4 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(cnn_attention_output, h4), b4)), self.relu_clip)
            layer_4 = tf.nn.dropout(layer_4, self.keep_dropout)
            print('没变形的layer_4', layer_4.shape)
            self.output = tf.reshape(layer_4, [self.batchSize, self.ksize, self.strides, 3])

        # Fully connected layer
        with tf.name_scope('fc6'):
            # 全连接层用于softmax分类
            nt_hpool = self.avg_pool_16x16(self.output)
            nt_hpool_flat = tf.reshape(nt_hpool, [self.batchSize, 3])
        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            #  不是GAN的损失
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nt_hpool_flat, labels=self.labels)
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            # 加l2正则化的损失函数，避免过拟合
            self.cost = tf.reduce_mean(cross_entropy) + self.l2_reg_lambda * self.l2_loss
        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.initial_learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        # Calculate accuracy
        with tf.name_scope('accuracy'):
            self.prediction = tf.argmax(nt_hpool_flat, 1, name="ylabel")
            self.actuals = tf.argmax(self.labels, 1)

            correct_pred = tf.equal(self.prediction, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # tp+fp+fn+tn=128
    """
    used to create a variable in CPU memory.
    """

    # 全局均值池化
    def avg_pool_16x16(self, x):
        return tf.nn.avg_pool(x, ksize=[1, self.ksize, self.ksize, 1], strides=[1, self.strides, self.strides, 1],
                              padding='SAME')

    def variable_on_cpu(self, name, shape, initializer):
        # Use the /cpu:0 device for scoped operations
        with tf.device('/cpu:0'):
            # Create or get apropos variable
            var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        return var

    # 进行单次训练
    def train_single_epoch(self, epoch_num=0):
        i = correct = total = loss = 0
        self.avg_acc = 0
        while True:
            self.keep_dropout = 0.98
            self.keep_prob = self.keep_prob
            self.is_training = True
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.myBatchGenerator.nextTrainBatch()
            # a = tf.reduce_sum(nextBatchLabels, 0)
            # print(nextBatch,nextBatchLabels)  # (128, 256) (128, 2)
            if len(nextBatch) * (i + 1) > len(self.x_train):
                break

            # self.session.run(self.train_op,
            #                  {self.input_data: nextBatch, self.cnn_input_data: nextBatch, self.labels: nextBatchLabels})
            acc, cost, _ = self.session.run([self.accuracy, self.cost,self.train_op],
                                         {self.cnn_input_data: nextBatch,self.labels: nextBatchLabels})
            # tp_op,fn_op,fp_op,tn_op, a = self.session.run([self.tp_op,self.fn_op,self.fp_op,self.tn_op,a],
            #                                           {self.input_data: nextBatch, self.cnn_input_data: nextBatch,self.labels: nextBatchLabels,
            #                                            })
            # Fit training using batch data
            # feeds = {self.input_data: nextBatch, self.labels: nextBatchLabels}
            # Write summary to Tensorboard
            if (i % 100 == 0):
                # acc, cost = self.session.run([self.accuracy, self.cost],
                #                                       {self.input_data: nextBatch, self.labels: nextBatchLabels,
                #                                        self.seq_len:seq_lengths})
                print("Iter " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(cost) + \
                      ", Training Accuracy= " + "{:.4f}".format(acc))

            correct += acc
            loss += cost
            total += len(nextBatch)
            i += 1

        total_accuracy = correct / (i)
        total_cost = loss / (i)
        self.history['acc'].append(total_accuracy)
        self.history['loss'].append(total_cost)

    # 训练使用 model2.train_epochs(5)->train_single_epoch
    def train_epochs(self, n_epochs):
        if not self.restore:
            tf.global_variables_initializer().run(session=self.session)

        num = 0
        while num < n_epochs:
            print("Epoch " + str(num + 1) + ":BIGRU_Att\n")
            self.train_single_epoch(num)
            # print("Average accuracy = " + "{:.3f}".format(self.avg_acc/4))
            self.test()

            if num % 1 == 0:
                self.save_model(num)
                print("saved to %s" % self.lstm_model_path)

            num += 1
        print('training finished.')

    # 测试使用 model2.test()
    def test(self):
        i = correct = total = loss = 0
        m_recall = m_precision = mf1_score = mm_accuracy = 0
        while True:
            # print('dropout jjj',self.dropOutRate)
            self.keep_prob = 1.0
            self.keep_dropout = 1.0
            self.is_training = False
            # print('dropout kkk',self.dropOutRate)

            nextBatch, nextBatchLabels = self.myBatchGenerator.nextTestBatch()
            if len(nextBatch) * (i + 1) > len(self.x_test):
                break

            acc, cost, yuce, zhenshi = self.session.run([self.accuracy, self.cost, self.prediction, self.actuals],
                                         {self.cnn_input_data: nextBatch,
                                          self.labels: nextBatchLabels})
            correct += acc
            loss += cost
            total += len(nextBatch)
            i += 1

        total_accuracy = correct / (i - 1)
        total_cost = loss / (i - 1)

        self.history['val_acc'].append(total_accuracy)
        self.history['val_loss'].append(total_cost)
        print("Minibatch Loss= " + "{:.4f}".format(total_cost) + ", Testing accuracy = " + "{:.4f}".format(
            total_accuracy))

    def save_model(self, step_num):
        self.saver = tf.train.Saver()
        self.saver.save(self.session,
                        self.lstm_model_path, global_step=step_num)
