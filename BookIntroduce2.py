import paddle
import pandas as pd
import paddle.nn as nn
import numpy as np
from paddle.io import Dataset
from sklearn.model_selection import train_test_split


# Step1:读取数据集

df = pd.read_csv('D:/MachineLearning/book-recommender/train_dataset.csv')
user_ids = df["user_id"].unique().tolist()

# 重新编码user 和 book，类似标签编码的过程
# 此步骤主要为减少id的编码空间
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

book_ids = df["item_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)}
book_encoded2book = {i: x for i, x in enumerate(book_ids)}

# 编码映射
df["user"] = df["user_id"].map(user2user_encoded)
df["book"] = df["item_id"].map(book2book_encoded)

num_users = len(user2user_encoded)
num_books = len(book_encoded2book)

user_book_dict = df.iloc[:].groupby(['user'])['book'].apply(list)

#Step2：构建负样本
# 随机挑选数据集作为负样本，负样本只需要对没有看的书籍进行随机采样
neg_df = []
book_set = set(list(book_encoded2book.keys()))
for user_idx in user_book_dict.index:
    book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
    book_idx = list(book_idx)
    neg_book_idx = np.random.choice(book_idx, 100)
    for x in neg_book_idx:
        neg_df.append([user_idx, x])

neg_df = pd.DataFrame(neg_df, columns=['user', 'book'])
neg_df['label'] = 0

# 正样本的标签
df['label'] = 1

# 正负样本合并为数据集
train_df = pd.concat([df[['user', 'book', 'label']],
                      neg_df[['user', 'book', 'label']]], axis=0)

train_df = train_df.sample(frac=1)
del df;

# 自定义数据集
#映射式(map-style)数据集需要继承paddle.io.Dataset
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode = 'train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)



x_train, x_val, y_train, y_val = train_test_split(train_df[['user', 'book']].values,
                                        train_df['label'].values.astype(np.float32).reshape(-1, 1))

train_dataset = SelfDefinedDataset(x_train, y_train)


# 测试数据集读取
for data, label in train_dataset:
    print(data.shape, label.shape)
    print(data, label)
    break

# 测试dataloder读取
train_loader = paddle.io.DataLoader(train_dataset, batch_size = 1280*4, shuffle = True)
for batch_id, data in enumerate(train_loader):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

val_dataset = SelfDefinedDataset(x_val, y_val)
val_loader = paddle.io.DataLoader(val_dataset, batch_size = 1280*4, shuffle = True)
for batch_id, data in enumerate(val_loader):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

EMBEDDING_SIZE = 32

# Step3:定义深度学习模型
class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)

        weight_attr_book = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.book_embedding = nn.Embedding(
            num_books,
            embedding_size,
            weight_attr=weight_attr_book
        )
        self.book_bias = nn.Embedding(num_books, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = paddle.dot(user_vector, book_vector)
        x = dot_user_book + user_bias + book_bias
        x = nn.functional.sigmoid(x)
        return x

model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)

model = paddle.Model(model)

# 定义模型损失函数、优化器和评价指标
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.003)
loss = nn.BCELoss()
metric = paddle.metric.Precision()

# # 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)

# 模型训练与验证
model.prepare(optimizer, loss, metric)
model.fit(train_loader, val_loader, epochs=5, save_dir='./checkpoints', verbose=1, callbacks=callback)

test_df = []
with open('D:/MachineLearning/book-recommender/submission.csv', 'w') as up:
    up.write('user_id,item_id\n')

# Step4:模型预测步骤
book_set = set(list(book_encoded2book.keys()))
for idx in range(int(len(user_book_dict) / 1000) + 1):
    # 对于所有的用户，需要预测其与其他书籍的打分
    test_user_idx = []
    test_book_idx = []
    for user_idx in user_book_dict.index[idx * 1000:(idx + 1) * 1000]:
        book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
        book_idx = list(book_idx)
        test_user_idx += [user_idx] * len(book_idx)
        test_book_idx += book_idx

    # 从剩余书籍中筛选出标签为正的样本
    test_data = np.array([test_user_idx, test_book_idx]).T
    test_dataset = SelfDefinedDataset(test_data, data_y=None, mode='predict')
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1280, shuffle=False)

    test_predict = model.predict(test_loader, batch_size=1024)
    test_predict = np.concatenate(test_predict[0], 0)

    test_data = pd.DataFrame(test_data, columns=['user', 'book'])
    test_data['label'] = test_predict
    for gp in test_data.groupby(['user']):
        with open('D:/MachineLearning/book-recommender/submission.csv', 'a') as up:
            u = gp[0]
            b = gp[1]['book'].iloc[gp[1]['label'].argmax()]
            up.write(f'{userencoded2user[u]}, {book_encoded2book[b]}\n')

del test_data, test_dataset, test_loader