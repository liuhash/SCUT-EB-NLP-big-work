import tensorflow as tf
import numpy as np
import os
import time
# 从硬盘或者网络连接读取文件存到的.keras\datasets下，这里是把数据集poetry.txt放到了C盘根目录下
path_to_file = tf.keras.utils.get_file("poetry.txt","file:///media/liunboyan/Data/pythonprojects/NLP/大作业/poetry.txt")
print(path_to_file)
# 读取文本内容
text = open(path_to_file, 'rb').read().decode(encoding='gbk')
# 打印出来
print(text) # 凭楼望北吟 诗为儒者禅 此格的惟仙 古雅如周颂 清和甚舜弦 冰生听瀑句 香发早梅篇……
# 列举文本中的非重复字符即字库
# 所有文本整理后，就是这么多不重复的字 ['龙', '龚', '龛', '龟'……]
vocab = sorted(set(text))
# 把这个字库保存到文件，以后使用直接拿，不用再去计算
np.save('vocab.npy',vocab)
# 创建从非重复字符到索引的映射
# 一个字典 {'龙': 1, '龚': 2, '龛': 3, '龟': 4……}，根据字能到数
char2idx = {u:i for i, u in enumerate(vocab)}

# 创建从索引到非重复字符的映射
idx2char = np.array(vocab) # 一个数组 ['龙' ... '龚' '龛' '龟']，根据数能找到字
# 将训练文件内容转换为索引的数据
# 全部文本转换为数字 [1020 4914 3146 ... 4731 2945    0]
text_as_int = np.array([char2idx[c] for c in text])

# 处理一句段文本，拆分为输入和输出两段
def split_input_target(chunk):
    input_text = chunk[:-1]  # 尾部去一个字
    target_text = chunk[1:]  # 头部去一个字
    return input_text, target_text  # 入：大江东去，出：大江东，江东去


# 创建训练样本，将转化为数字的诗句外面套一层壳子，原来是[x]
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# 所有样本中，每24个字作为一组
sequences = char_dataset.batch(24, drop_remainder=True)  # 数据当前状态：((24,x))
# 将每24个字作为一组所有样本，掐头去尾转为输入，输出结对
dataset = sequences.map(split_input_target)  # 数据当前状态：((23,x), (23,x))

# 将众多输入输出对打散，并64个为一组
BATCH_SIZE = 64
# 数据当前状态：((64, 23), (64, 23))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
# 获取一批训练的输入，输出
train_batch, train_labels = next(iter(dataset))
# 构建一个模型的方法
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
    return model

# 词集的长度,也就是字典的大小
vocab_size = len(vocab)
# 嵌入的维度，也就是生成的embedding的维数
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024

# 整一个模型
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# 损失函数
def loss(labels, logits):
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# 配置优化器和损失函数
model.compile(optimizer='adam', loss=loss)
# 训练结果保存的目录
checkpoint_dir = './training_checkpoints'
# 文件名 ckpt_训练轮数
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# 训练的回调
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
# 进行训练
history = model.fit(dataset, epochs=20, callbacks=[checkpoint_callback])