import numpy as np
import pandas as pd

train_df = pd.read_csv('./90k.csv', index_col=None)
test_df = pd.read_csv('./10k.csv', index_col=None)

## 训练集中有 354 个数据没有微博内容
data_df_na = train_df[train_df['微博中文内容'].isna()]
# print(data_df_na)
## 查看data_df_na 的标签分布,标签为0的占93.79%
print(data_df_na['情感倾向'].value_counts())
## 取微博内容做训练数据
data_df_not_na = train_df[train_df['微博中文内容'].notna()]
# print(data_df_not_na)
## 查看标签分布
print(data_df_not_na['情感倾向'].value_counts())



## 训练集中有 354 个数据没有微博内容
data_df_na = test_df[test_df['微博中文内容'].isna()]
# print(data_df_na)
## 查看data_df_na 的标签分布,标签为0的占93.79%
print(data_df_na['情感倾向'].value_counts())
## 取微博内容做训练数据
data_df_not_na = test_df[test_df['微博中文内容'].notna()]
# print(data_df_not_na)
## 查看标签分布
print(data_df_not_na['情感倾向'].value_counts())