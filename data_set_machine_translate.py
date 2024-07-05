import paddle


# 通过继承Data.Dataset来创建自定义的数据集
class MyDataSet(paddle.io.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        """
        返回数据集中的样本数
        """
        # 第一个维度就是batch_size，也就是样本总数
        # paddle中获取形状用shape[下标]
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        # 根据样本的索引idx返回一个对应的样本及其标签
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# S: Symbol that shows starting of decoding input
# E: Symbol that shows ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is shorter than time steps
# 第一个维度是每个样本；第二个维度的：
#   第一个元素是训练时编码器的输入；第二个元素是训练时解码器的输入；第三个元素是解码器的标签，相当于训练时句子的ground truth。
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 源语言词典，pad对应的索引必须是0
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
# 源语言词语空间大小，实际情况下，它的长度应该是所有德语单词的个数
src_vocab_size = len(src_vocab)

# 目标语言词典
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# 词典，下标->词
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# 目标语言词语空间大小，实际情况下，它应该是所有英语单词个数
tgt_vocab_size = len(tgt_vocab)

# enc_input max sequence length，源语言的句子seq_len
src_len = 5
# dec_input(=dec_output) max sequence length，目标语言的句子seq_len
tgt_len = 6

BATCHSIZE=2


# 根据原始的句子列表，生成数据
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # 遍历每一个样本
    for i in range(len(sentences)):
        # 当前样本中，源语言句子的每个词的索引，放在enc_input中，作为编码器的输入
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        # 当前样本中，目标语言的每个词的索引，放在dec_input中
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        # 当前样本中，目标语言的每个词的索引，放在dec_input中
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        # Python list的expand：将一个可迭代对象（如另一个列表、元组、集合、字符串等）中的所有元素逐个添加到列表的末尾
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    # 注意要转成LongTensor，因为是索引，整型才能在EmbeddingLayer中查参数矩阵
    # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
    # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
    # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
    return (paddle.to_tensor(enc_inputs, dtype='int64'),
            paddle.to_tensor(dec_inputs, dtype='int64'),
            paddle.to_tensor(dec_outputs, dtype='int64'))


# 获取样本数据的DataLoader
def data_loader():
    # shape：[源语言全体样本数, src_len]，[目标语言全体样本数, tgt_len]，[目标语言全体样本数, tgt_len]
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = paddle.io.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)
    return loader

