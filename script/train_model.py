import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import gensim
import pandas as pd

MAX_SEQUENCE_LENGTH = 1000 # 每篇文章选取1000个词
MAX_NB_WORDS = 10000 # 将字典设置为含有1万个词
EMBEDDING_DIM = 200 # 词向量维度
VALIDATION_SPLIT = 0.5 # 测试集大小

# STEP 1
# 得到一份字典(embeddings_index)
embeddings_index = {}

print('Indexing word vectors.')
if os.path.exists('f.model'):     # 判断文件是否存在
    model = gensim.models.Word2Vec.load('f.model')
else:
    print('Model not found.')
word_vectors = model.wv
for word, vocab_obj in model.wv.vocab.items():
    if int(vocab_obj.index) < MAX_NB_WORDS:
        embeddings_index[word] = word_vectors[word]
del model, word_vectors # 删掉gensim模型释放内存
print('Found %s word vectors.' % len(embeddings_index))

# print out:
# Indexing word vectors.
# Found 10000 word vectors.

# STEP 2
print('Processing text dataset')

texts = []  # list of text samples
labels = []  # list of label ids

# 读取数据
data = pd.read_excel('data.xlsx')

#提取内容和标签
texts = data['content'].values.tolist()
labels = data['label'].values.tolist()
del data

print('Found %s texts.' % len(texts))

# print out
# Processing text dataset
# Found 6946 texts.

# STEP 3
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts) # 传入训练数据，得到训练数据中出现的词的字典
sequences = tokenizer.texts_to_sequences(texts) # 根据训练数据中出现的词的字典，将训练数据转换为sequences

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 限制每篇文章的长度

labels = to_categorical(np.asarray(labels)) # label one hot表示
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# print out
# Found 100323 unique tokens.
# Shape of data tensor: (6946, 1000)
# Shape of label tensor: (6946, 2) # 我的文本类别有2类

# STEP 4
# 准备训练集和测试集

# 打乱文本顺序
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# 切割数据
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# STEP 5
# 准备embedding layer

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
        embedding_matrix[i] = embedding_vector

# 将预训练好的词向量加载如embedding layer
# 我们设置 trainable = False，代表词向量不作为参数进行更新
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# STEP 6
# 训练模型
# 训练  1D 卷积神经网络 使用 Maxpooling1D
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_sequences)
x = MaxPooling1D(pool_size=5)(x)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = MaxPooling1D(pool_size=5)(x)
x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = MaxPooling1D(pool_size=35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# 如果希望短一些时间，epochs调小
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=1,
          batch_size=128)

score, acc = model.evaluate(x_val, y_val)
print('Test score:',score)
print('Test accuracy:', acc)

model.save('1dcnn_model.h5')