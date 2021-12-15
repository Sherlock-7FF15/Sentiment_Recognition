import numpy as np
import random


class BatchGenerator:
    def __init__(self, X_train, y_train, X_test, y_test, batchSize, shuffle=True):
        # (16000, 256) (4000, 256) (16000,) (4000,)
        self.X_train = X_train  # (16000, 256)
        self.y_train = y_train  # (16000,)
        self.X_test = X_test  # (4000, 256)
        self.y_test = y_test  # (4000,)
        self.X_train_offset = 0
        self.X_test_offset = 0
        self.shuffle = shuffle
        self.batchSize = batchSize  # 128

    def shuffleIfTrue(self, name):
        if name == 'train':
            # 训练集打乱
            arr_train = np.arange(len(self.X_train))
            np.random.shuffle(arr_train)

            self.X_train = self.X_train[arr_train]
            self.y_train = self.y_train[arr_train]
        elif name == 'test':
            # 测试集打乱
            arr_test = np.arange(len(self.X_test))
            np.random.shuffle(arr_test)
            self.X_test = self.X_test[arr_test]
            self.y_test = self.y_test[arr_test]

    def nextTrainBatch(self):
        start = self.X_train_offset  # 0,128,256,384,512,640,768,896,1024
        end = self.X_train_offset + self.batchSize  # 0+128,128+128,256+128,384+128,512+128,640+128,768+128,896+128

        self.X_train_offset = end  # 128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920

        # handle wrap around    ,
        if end > len(self.X_train):  # 比如len(self.X_train)=1000
            # print('end..',end)
            spillover = end - len(self.X_train)  # 1024-1000=24
            # print('spillover....',spillover)
            self.X_train_offset = spillover  # 24
            # print('开始',start)
            X = np.concatenate((self.X_train[start:], self.X_train[:spillover]), axis=0)
            # lengths = np.array([128 for s in np.concatenate((seq_message[start:], seq_message[:spillover]), axis=0)])

            a = []
            for i in self.y_train[start:]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            for i in self.y_train[:spillover]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            y = np.array(a)

            # y = np.transpose([np.concatenate((self.y_train[start:], self.y_train[:spillover]), axis=0),
            #                    1 - np.concatenate((self.y_train[start:], self.y_train[:spillover]), axis=0)])
            # sparse_labels = self.sparse_tuple_from_label(y)
            # print('shi',y.shape)


            self.X_train_offset = 0
            self.shuffleIfTrue('train')

        else:
            # [0:128] [128:256] [256:384] [384:512] [512:640] [640:768] [768:896] [1024:1152]
            X = self.X_train[start:end]
            # lengths = np.array([128 for s in seq_message[start:end]],dtype=np.int32)

            # print('----',self.y_train[start:end])
            # y = np.transpose([self.y_train[start:end], 1 - self.y_train[start:end]])
            # print('no',y.shape)
            # sparse_labels = self.sparse_tuple_from_label(y)

            a = []
            for i in self.y_train[start:end]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            y = np.array(a)

        X = X.astype(np.int32, copy=False)
        # y = y.astype(np.float32, copy=False)

        return X, y

    def nextTestBatch(self):
        start = self.X_test_offset
        end = self.X_test_offset + self.batchSize
        self.X_test_offset = end

        # handle wrap around    ,
        if end > len(self.X_test):
            spillover = end - len(self.X_test)
            self.X_test_offset = spillover
            X = np.concatenate((self.X_test[start:], self.X_test[:spillover]), axis=0)
            # lengths = np.array([128 for s in np.concatenate((seq_message[start:], seq_message[:spillover]), axis=0)])

            a = []
            for i in self.y_test[start:]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            for i in self.y_test[:spillover]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            y = np.array(a)
            # y = np.transpose([np.concatenate((self.y_test[start:], self.y_test[:spillover]), axis=0),
                             # 1 - np.concatenate((self.y_test[start:], self.y_test[:spillover]), axis=0)])
            # sparse_labels = self.sparse_tuple_from_label(y)

            self.X_test_offset = 0

            self.shuffleIfTrue('test')

        else:
            X = self.X_test[start:end]
            # lengths = np.array([128 for s in seq_message[start:end]],dtype=np.int32)

            # y = np.transpose([self.y_test[start:end], 1 - self.y_test[start:end]])

            a = []
            for i in self.y_test[start:end]:
                if i == -1:
                    a.append([1, 0, 0])
                elif i == 0:
                    a.append([0, 1, 0])
                elif i == 1:
                    a.append([0, 0, 1])
            y = np.array(a)
            # sparse_labels = self.sparse_tuple_from_label(y)

        X = X.astype(np.int32, copy=False)
        y = y.astype(np.float32, copy=False)

        return X, y


