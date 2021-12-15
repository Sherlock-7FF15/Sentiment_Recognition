import matplotlib as mpl
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties

# 设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

from lib_utils import model_one_layer_BIGRU
from lib_utils import model_BIGRU_Att
from lib_utils import model_BIGRU_CNN_Att
from lib_utils import model_two_layer_BIGRU
from lib_utils import model_CNN_Att
from lib_utils import model_CNN
from lib_utils import model_CNN_BIGRU_Att

from lstm_train import loadfile, tokenizer, word2vec_train, get_data, \
    plot_with_labels, analysisfile


def measurement(model):
    # bar_graph = []
    average_accuracy = []
    average_recall = []
    max_F1_score = []
    loss = []
    a6_list = model.history['f1_score']
    #
    recall_list_6 = model.history['recall']
    precision_list_6 = model.history['precision']
    recall_6 = sorted(recall_list_6, reverse=False)
    precision_6 = sorted(precision_list_6, reverse=True)
    #
    a6max_f1_score = max(model.history['f1_score'])
    print('a6最大的f1值', a6max_f1_score)
    a6recall_list = model.history['recall'][-1]
    a6precision_list = model.history['precision'][-1]
    #
    average_accuracy.append(a6precision_list)
    average_recall.append(a6recall_list)
    max_F1_score.append(a6max_f1_score)

    a6loss = model.history['val_loss']


def one_layer_BIGRU(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第1种模型model_one_layer_BIGRU---------')
    model1 = model_one_layer_BIGRU.LstmTFModel(useAttention=True, restore=False,
                                               index_dict=index_dict, word_vectors=word_vectors,
                                               combined=combined,
                                               y=y, embedding_weights=embedding_weights)
    model1.train_epochs(epoch)
    model1.test()
    # 衡量指标
    measurement(model1)


def BIGRU_CNN_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第2种模型BIGRU_CNN_Att---------')
    model2 = model_BIGRU_CNN_Att.LstmTFModel(useAttention=True, restore=False,
                                             index_dict=index_dict,
                                             word_vectors=word_vectors, combined=combined, y=y,
                                             embedding_weights=embedding_weights)
    model2.train_epochs(epoch)
    model2.test()
    # 衡量指标
    measurement(model2)


def BIGRU_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第3种模型BIGRU_Att--------')
    model3 = model_BIGRU_Att.LstmTFModel(useAttention=True, restore=False, index_dict=index_dict,
                                         word_vectors=word_vectors,
                                         combined=combined,
                                         y=y, embedding_weights=embedding_weights)
    model3.train_epochs(epoch)
    model3.test()
    # 衡量指标
    measurement(model3)


def two_layer_BIGRU(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第4种模型two_layer_BIGRU--------')
    model4 = model_two_layer_BIGRU.LstmTFModel(useAttention=True, restore=False,
                                               index_dict=index_dict,
                                               word_vectors=word_vectors, combined=combined,
                                               y=y, embedding_weights=embedding_weights)
    model4.train_epochs(epoch)
    model4.test()
    # 衡量指标
    measurement(model4)


def CNN_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第5种模型CNN_Att--------')
    model5 = model_CNN_Att.LstmTFModel(useAttention=True, restore=False,
                                       index_dict=index_dict,
                                       word_vectors=word_vectors, combined=combined,
                                       y=y, embedding_weights=embedding_weights)
    model5.train_epochs(epoch)
    model5.test()
    # 衡量指标
    measurement(model5)


def one_CNN(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第六种模型CNN--------')
    model6 = model_CNN.LstmTFModel(useAttention=True, restore=False, index_dict=index_dict,
                                   word_vectors=word_vectors,
                                   combined=combined,
                                   y=y, embedding_weights=embedding_weights)
    model6.train_epochs(epoch)
    model6.test()
    measurement(model6)

def CNN_BIGRU_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第7种模型model_CNN_BIGRU_Att---------')
    model7 = model_CNN_BIGRU_Att.LstmTFModel(useAttention=True, restore=False,
                                               index_dict=index_dict, word_vectors=word_vectors,
                                               combined=combined,
                                               y=y, embedding_weights=embedding_weights)
    model7.train_epochs(epoch)
    model7.test()
    # 衡量指标
    measurement(model7)

def new_mm_model(index_dict, word_vectors, combined, y, embedding_weights, epoch):
    print('-------第8种模型model_CNN_BIGRU_Att---------')
    model8 = model_BIGRU_MAtt.LstmTFModel(useAttention=True, restore=False,
                                          index_dict=index_dict, word_vectors=word_vectors,
                                          combined=combined,
                                          y=y, embedding_weights=embedding_weights)
    model8.train_epochs(epoch)
    model8.test()
    # 衡量指标
    measurement(model8)


if __name__ == '__main__':
    # Run data prep routine if some files are not found
    # 训练模型，并保存
    print('Loading Data...')
    combined, y = loadfile()  # 读取pos、neg文本，两个文本长度总和
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)  # 将文本去除停用词，分词
    # analysisfile(combined)  #文本长度-频次图

    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    # print('index_dict', index_dict)
    # print('词向量',word_vectors[','])

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)

    # print("x_train.shape and y_train.shape:")
    print('*****',x_train.shape, y_train.shape, n_symbols)
    ##########################################################
    # labels,messages = initialize_inputs()
    epoch = 14
    # 第一个
    # one_layer_BIGRU(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第二个
    BIGRU_CNN_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第三个
    # BIGRU_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第四个
    # two_layer_BIGRU(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第五个
    # CNN_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第六个
    # one_CNN(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第七个
    # CNN_BIGRU_Att(index_dict, word_vectors, combined, y, embedding_weights, epoch)
    # 第八个
    # new_mm_model(index_dict, word_vectors, combined, y, embedding_weights, epoch)

