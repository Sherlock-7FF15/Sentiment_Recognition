import pandas as pd

train_df = pd.read_csv('./chn_test.csv', index_col=None)

## 取微博内容做训练数据
data_df_not_na = train_df[train_df['微博中文内容'].notna()]
print(data_df_not_na)
## 查看标签分布
print(data_df_not_na['情感倾向'].value_counts())
