import pandas as pd
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
import numpy as np

from lstm_train import loadfile, tokenizer, word2vec_train, get_data, \
                        plot_with_labels, analysisfile,create_dictionaries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def loadvalid():
    content = pd.read_csv('train_ dataset/10k.csv',
                          header=None, index_col=None)
    # 处理文本
    a = []
    for k, v in enumerate(content[3][1:]):
        a.append(v)
    combined = np.array(a)

    return combined


def create_csv(data):
    # 任意的多组列表
    content = pd.read_csv('train_ dataset/10k.csv',
                          header=None, index_col=None)
    # 获得验证文本id
    a = []
    for k, v in enumerate(content[0][1:]):
        a.append(int(str(v).strip()))
    id = a
    y = data[:10004]
    print('两个长度',len(id),len(y))
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'id': id, 'y': y})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("result.csv", index=False, sep=',')

class Bactchgenerate():
    def __init__(self):
        self.X_test_offset = 0
        self.batchSize = 128

    def nextValidBatch(self,valid_data):

        start = self.X_test_offset
        end = self.X_test_offset + self.batchSize
        self.X_test_offset = end
        if end > len(valid_data):  # 比如len(self.X_train)=1000
            spillover = end - len(valid_data)
            self.X_test_offset = spillover
            X = np.concatenate((valid_data[start:], valid_data[:spillover]), axis=0)
        else:
            ###
            X = valid_data[start:end]
        X = X.astype(np.int32, copy=False)
        return X

# 重新加载模型
def load_model(txtdata):
    sess = tf.Session()
    X = None  # input
    yhat = None  # output
    """
        Loading the pre-trained model and parameters.
    """
    modelpath = './model/pretrained_lstm_tf.model'
    saver = tf.train.import_meta_graph(modelpath + '-5.meta')
    # saver.restore(sess, tf.train.latest_checkpoint(modelpath))
    saver.restore(sess, tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("cnn_input_data:0")
    yhat = graph.get_tensor_by_name("accuracy/ylabel:0")
    print('Successfully load the pre-trained model!')

    data = np.array(txtdata)
    # 循环加载验证数据
    ret = []
    i = 0
    batchgenerate = Bactchgenerate()
    while True:
        print('第' + str(i + 1) + '批' + '==========================================')
        # 先取128批次数据
        nextBatch = batchgenerate.nextValidBatch(data)
        if len(nextBatch) * (i + 1) > len(data):
            output = sess.run(yhat, feed_dict={X: nextBatch})  #
            output = list(output)
            ret += output
            break

        output = sess.run(yhat, feed_dict={X: nextBatch})  #
        # output = output.reshape(-1, 1)
        # ret.append(output.tolist())
        output = list(output)
        ret += output
        print('预测输出',ret)
        i += 1
    print('最后的结果',ret)
    return ret
# 验证模型
# 加载验证数据集
valid_data = loadvalid()
combined = tokenizer(valid_data)
model = Word2Vec.load('lstm_data_test/Word2vec_model.pkl')
_, _, combined = create_dictionaries(model,combined)
a = []
# 获得预测结果
predict_result = load_model(combined)
labels = [-1, 0, 1]
ret = [labels[i] for i in predict_result]
# # 生成csv文件
create_csv(ret)
