from numpy.core.records import array
import tensorflow as tf
import numpy as np
import os
import time
import random

# 读取字典
vocab = np.load('vocab.npy')
# 创建从非重复字符到索引的映射
char2idx = {u: i for i, u in enumerate(vocab)}
print(char2idx)
# 创建从数字到字符的映射
idx2char = np.array(vocab)
# 词集的长度,也就是字典的大小
vocab_size = len(vocab)
# 嵌入的维度，也就是生成的embedding的维数
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024


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


# # 读取保存的训练结果
# checkpoint_dir = './training_checkpoints'
# tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim,
                    rnn_units, batch_size=1)
# # 当初只保存了权重，现在只加载权重
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# # 从历史结果构建起一个model
# model.build(tf.TensorShape([1, None]))
# start_string = "大"
# # # 将起始字符串转换为数字
# input_eval = [char2idx[s] for s in start_string]
# # print(input_eval)  # [1808]
# # # 训练模型结构一般是多套输入多套输出，要升维
# input_eval = tf.expand_dims(input_eval, 0)
# # print(input_eval)  # Tensor([[1808]])
#
# # 获得预测结果，结果是多维的
# predictions = model(input_eval)
# print(predictions)
# '''
# 输出的是预测结果，总共输入'明'一个字，输出分别对应的下一个字的概率，总共有5380个字
# shape=(1, 1, 5380)
# tf.Tensor(
# [[[ -3.3992984    2.3124864   -2.7357426  ... -10.154563 ]]])
# '''
#
# # 预测结果，删除批次的维度[[xx]]变为[xx]
# predictions1 = tf.squeeze(predictions, 0)
# # 用分类分布预测模型返回的字符，从5380个字中根据概率找出num_samples个字
# predicted_ids = tf.random.categorical(predictions1, num_samples=1).numpy()
# print(idx2char[predicted_ids])  # [['名']]


# 根据一段文本，预测下一段文本
def generate_text(model, start_string, num_generate=6):
    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    # 上面结果是[2,3,4,5]

    # 训练模型结构一般是多套输入多套输出，要升维
    input_eval = tf.expand_dims(input_eval, 0)
    # 上结果变为[[2,3,4,5]]

    # 空字符串用于存储结果
    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        # 获得预测结果，结果是多维的
        predictions = model(input_eval)
        # 预测结果，删除批次的维度[[xx,xx]]变为[xx,xx]
        predictions = tf.squeeze(predictions, 0)
        # 用分类分布预测模型返回的字符
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)
        # 将预测的字符存起来
        text_generated.append(idx2char[predicted_id])

    # 最终返回结果
    return start_string + ''.join(text_generated)


# %%
s = "掘金不止"
array_keys = list(s)
all_string = ""
for word in array_keys:
    all_string = all_string+" "+ word
    next_len = 5 - len(word)
    print("input:", all_string)
    all_string = generate_text(model, start_string=all_string, num_generate=next_len)
    print("out:", all_string)

print("最终输出:" + all_string)
