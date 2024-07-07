import math

import paddle
import numpy as np

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


class Config:
    def __init__(self, src_vocab_size, tgt_vocab_size):
        """transformer的超参数"""
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size


def get_attn_subsequent_mask(seq):
    """
    遮盖decoder输入的未来词
    :param seq:decoder中的句子，形状 [batch_size, tgt_len]。
    :return:掩码张量，形状 [batch_size, tgt_len, tgt_len]。
    """
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    # 此时subsequence_mask是一个形状为[batch_size,tgt_len,tgt_len]的上三角矩阵，对角线之上的元素值为1.
    subsequence_mask = paddle.to_tensor(subsequence_mask,dtype=paddle.bool)
    return subsequence_mask


def get_attn_pad_mask(seq_q:paddle.Tensor, seq_k:paddle.Tensor):
    """
    向量q关于向量k的掩码矩阵
    :param seq_q:形状为[batch_size , len_q]。
    :param seq_k:形状为[batch_size , len_k]。
    :return:形状为[batch_size, len_q, len_k]，如果seq_k的第k列为pad，那么输出的张量中[:,:,k]是True。
    """
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    # 查找的是seq_k中的0
    pad_attn_mask = paddle.equal(seq_k,paddle.zeros(seq_k.shape,dtype=paddle.int64)).unsqueeze(1)  # [batch_size, 1, len_k], True is masked

    # 扩张到len_q，expand和torch写法不一样，位置参数是list|tuple
    return pad_attn_mask.expand(shape=[batch_size, len_q, len_k])  # [batch_size, len_q, len_k]


class PositionalEncoding(paddle.nn.Layer):
    """
    位置编码
    """

    def __init__(self,dropout=0.5, max_len=5000):
        """
        :param dropout: 位置编码随机置0的概率
        :param max_len: 句子的最大长度，也就是词个数。
        """
        super().__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)

        # pe：形状为[max_len,d_model]的全为0的tensor
        pe = paddle.zeros([max_len, d_model])

        # position:将原本形状为[max_len]的张量转换成[max_len,1]，即[5000,1]。
        # 这里插入一个维度是为了后面能够进行广播机制然后和div_term直接相乘
        position = paddle.arange(0, max_len,dtype=paddle.float32).unsqueeze(1)

        # 利用指数函数e和对数函数log取下来，方便计算。
        # div_term里的每个元素表示的就是原论文中的sin(A*x)、cons(A*x)中的系数A。
        div_term = paddle.exp(paddle.arange(0, d_model, 2) *
                              -(math.log(10000.0) / d_model))

        # 因为div_term的形状为[d_model/2]，即[256]，符合广播条件，广播后两个tensor形状都会变成[5000,256]。
        # *表示两个tensor对应位置处的两个元素相乘

        # 把pe中全部行的偶数列赋值
        pe[:, 0::2] = paddle.sin(position * div_term)

        # 把pe中全部行的奇数列赋值
        pe[:, 1::2] = paddle.cos(position * div_term)

        # 重塑形状为 (max_len, 1, d_model)，以便与编码器/解码器的输入数据X的形状匹配。
        # transpose写法和torch不一样，必须给出转换后的全部维度的位置，而非互换的两个维度
        pe = pe.unsqueeze(0).transpose(perm=[1,0,2])
        # 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入x: [seq_len, batch_size, d_model]
        # 这里的self.pe是从缓冲区里拿的。切片操作，把pe第1维的前seq_len个tensor和x相加，其他维度不变
        # 这里其实也有广播机制，pe:[max_len,1,d_model]，第二维大小为1，会自动扩张到batch_size大小，实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.shape[0], :,:]
        # 位置编码也dropout
        return self.dropout(input=x)


class ScaledDotProductAttention(paddle.nn.Layer):
    """
    计算多头Z，softmax((QK^T)/sqrt(d_k)) V
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q: paddle.Tensor, K: paddle.Tensor, V: paddle.Tensor, attn_mask: paddle.Tensor):
        """
        根据Q关于K的attention，计算对应的V
        :param Q: [batch_size , n_heads , len_q , d_k]
        :param K: [batch_size , n_heads , len_k , d_k]
        :param V: [batch_size , n_heads , len_k , d_v]
        :param attn_mask: q关于k的掩码，要求和(QK^T)的形状必须一致，才能对应元素相加。
        :return:
        """
        # 矩阵乘法是对张量的最后两个维度做乘法
        scores = paddle.matmul(Q, K.transpose([0,1,3,2])) / np.sqrt(d_k)
        # scores：[batch_size , n_heads , len_q , len_k]

        # 要求attn_mask和scores的形状一致

        # 把被attn_mask的地方置为无穷小，softmax之后会趋近于0，Q会忽视这部分的权重
        paddle.masked_fill(scores, attn_mask, -1e9)

        # 在最后一个维度上计算softmax，attn:[batch_size,n_heads,len_q,len_k]
        attn = paddle.nn.Softmax(axis=-1)(scores)

        # 和V矩阵相乘，context:[batch_size,n_heads,len_q,d_v]
        context = paddle.matmul(attn, V)

        return context, attn


class MultiHeadAttention(paddle.nn.Layer):
    """
    一个完整的multi-head attention
    """

    def __init__(self):
        super().__init__()
        # Wq,Wk,Wv其实就是一个线性层，用来将输入映射为Q、K、V
        # 注意理解这里的输出是d_k * n_heads，因为是先映射，后分头。
        self.W_Q = paddle.nn.Linear(d_model, d_k * n_heads)
        self.W_K = paddle.nn.Linear(d_model, d_k * n_heads)
        self.W_V = paddle.nn.Linear(d_model, d_v * n_heads)

        # W_O，用于把拼接起来的Z映射回d_model维。
        self.linear = paddle.nn.Linear(n_heads * d_v, d_model)

        # layer norm层
        self.layer_norm = paddle.nn.LayerNorm(d_model)

    def forward(self, input_Q: paddle.Tensor, input_K: paddle.Tensor, input_V: paddle.Tensor, attn_mask: paddle.Tensor):
        '''
        一个multi-head attention的前向传播，每个multi-head attention的输入输出形状一致。
        :param input_Q: [batch_size , len_q , d_model]
        :param input_K:[batch_size , len_k , d_model]
        :param input_V:[batch_size , len_k , d_model]
        :param attn_mask:[batch_size , len_q , len_k]，向量q关于k的掩码
        :return: [batch_size , len_q , d_model]
        '''

        # 保留原数据做Add
        # paddle中size写法与torch不一样，size是全部数据的大小，而非形状，paddle中使用形状要用shape
        residual, batch_size = input_Q, input_Q.shape[0]

        # 分头；一定要注意的是q和k分头之后维度是一致的，所以一看这里都是d_k
        # view写法与torch不一致，PyTorch 参数 shape 既可以是可变参数，也可以是 list/tuple/torch.Size/dtype 的形式， Paddle 参数 shape_or_dtype 为 list/tuple/dtype 的形式。
        # 对于可变参数的用法，需要进行转写。
        # Q: [batch_size , n_heads , len_q , d_k]
        Q = self.W_Q(input_Q).reshape([batch_size, -1, n_heads, d_k]).transpose([0,2,1,3])
        # K: [batch_size , n_heads , len_k , d_k]
        K = self.W_K(input_K).reshape([batch_size, -1, n_heads, d_k]).transpose([0,2,1,3])
        # V: [batch_size , n_heads , len_k , d_v]
        V = self.W_V(input_V).reshape([batch_size, -1, n_heads, d_v]).transpose([0,2,1,3])

        # attn_mask:[batch_size , len_q , len_k] ---> [batch_size , n_heads , len_q , len_k]
        # 就是把掩码信息复制n份，重复到n个头上以便计算多头注意力机制的掩码
        attn_mask = attn_mask.cast(dtype=paddle.int64).unsqueeze(1).repeat_interleave(n_heads, 1).cast(dtype=paddle.bool)

        # 计算ScaledDotProductAttention
        # 得到的结果有两个：context: [batch_size , n_heads , len_q , d_v],
        # attn: [batch_size , n_heads , len_q , len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # 这里实际上在拼接n个头，把n个头的加权注意力输出拼接，使每个q向量有一个长度为n_heads * d_v的表征向量
        #  [batch_size , n_heads , len_q , d_v]->[batch_size , len_q, n_heads*d_v]。
        context = context.transpose([0,2, 1,3]).reshape([batch_size, -1, n_heads * d_v])
        # 通过线性层映射回输入维度，output: [batch_size , len_q , d_model]
        output = self.linear(context)

        # 过残差、LN，输出output: [batch_size , len_q , d_model]和这一层的加权注意力表征向量
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(paddle.nn.Layer):
    """
    两层前馈神经网络
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(d_model, d_ff, bias_attr=False),  # 没有偏置
            paddle.nn.ReLU(),
            paddle.nn.Linear(d_ff, d_model, bias_attr=False)  # 没有偏置
        )

    def forward(self, inputs):
        """
        一次前馈神经网络的前向传播
        :param inputs: [batch_size, seq_len, d_model]，这一层输入就是multi-head self-attention的输出
        :return: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        #  在FF里面做了Add&LayerNorm，之后再输出 [batch_size, seq_len, d_model]
        return paddle.nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(paddle.nn.Layer):
    """
    一个完整的Encoder Layer
    """

    def __init__(self):
        super().__init__()
        # 一个multi-head self-attention
        self.enc_self_attn = MultiHeadAttention()
        # FF层
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        一次Encoder Layer的前向传播
        :param enc_inputs:[batch_size, src_len, d_model]，某种表征：要么是原始训练数据句子embedding；要么是上一个encoder layer的输出。
        :param enc_self_attn_mask:[batch_size, src_len, src_len]，编码器的pad 掩码
        :return:
            enc_outputs，多头注意力处理后的输出，形状仍为 [batch_size, src_len, d_model]
            attn 是注意力权重矩阵，形状为 [batch_size, n_heads, src_len, src_len]
        """
        # 需要注意的是enc_self_attn的参数input_Q, input_K, input_V在编码器中全都是enc_inputs
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        # FF
        enc_outputs = self.pos_ffn(enc_outputs)

        # enc_outputs，它又会作为下一层encoder layer的输入
        return enc_outputs, attn


class Encoder(paddle.nn.Layer):
    def __init__(self, src_vocab_size: int):
        super(Encoder, self).__init__()
        # 这行其实就是生成一个参数矩阵，大小是: src_vocab_size * d_model，矩阵的每一行就是一个单词的embedding
        self.src_emb = paddle.nn.Embedding(src_vocab_size, d_model)
        # 位置编码，这里是固定的，传入d_model这个数立刻就生成了对应位置的编码
        self.pos_emb = PositionalEncoding()
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以不需要重复走nn.Embedding和PositionalEncoding。
        self.layers = paddle.nn.LayerList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        Encoder部分包含三个部分：词向量embedding（处理原始输入）；位置编码部分（处理原始输入）；N个encoder layer。
        :param enc_inputs: [batch_size, src_len]，源语言，一批次中有 batch_size 个序列，每个序列长度为 src_len。
        :return:  enc_outputs：最后一层encoder layer的输出[batch_size, src_len, d_model]，注意它一定是和第一层的输入形状一样。
                  enc_self_attns 存储了每一层的注意力权重，这可以用于分析模型是如何分配注意力的。
        """
        # enc_outputs：[batch_size, src_len, d_model],其实就是根据单词索引查Embedding参数矩阵的行
        enc_outputs = self.src_emb(enc_inputs)

        # enc_outputs：[batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose([ 1,0,2])).transpose([ 1,0,2])

        # 根据句子的长度获取掩码，掩码的意思是句子q关于句子k的掩码，这里q和k都是源语言的句子，所以长度一样都是src_len。enc_self_attn_mask：[batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # 记录起来每个encoder layer的输出
            enc_self_attns.append(enc_self_attn)
        # enc_outputs：最后一层encoder layer的输出[batch_size, src_len, d_model]，注意它一定是和第一层的输入形状一样。
        return enc_outputs, enc_self_attns


class DecoderLayer(paddle.nn.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # self-attention
        self.dec_self_attn = MultiHeadAttention()
        # cross-attention
        self.dec_enc_attn = MultiHeadAttention()
        # FF
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs:  [batch_size, tgt_len, d_model]
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len],tgt关于tgt的掩码，等于未来词mask+pad mask
        :param dec_enc_attn_mask: [batch_size, tgt_len, src_len]，tgt关于src的pad掩码（源语言里为pad的词，tgt中的词关于它的attention是负无穷）
        :return:
            dec_outputs:[batch_size, tgt_len, d_model]
        """
        # 第一个attention，传入参数input_Q input_K input_V都是dec_inputs，掩码是tgt关于tgt的mask=未来mask+pad mask
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # 第二个attention，input_Q是上一层self.dec_self_attn的输出dec_outputs；input_K和input_V是Encoder的最终输出enc_outputs；掩码是tgt关于src的pad mask
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # FF
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]

        # dec_outputs:[batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(paddle.nn.Layer):
    def __init__(self, tgt_vocab_size: int):
        super(Decoder, self).__init__()
        # word embedding 参数矩阵
        self.tgt_emb = paddle.nn.Embedding(tgt_vocab_size, d_model)
        # 生成位置编码
        self.pos_emb = PositionalEncoding()
        # 叠加若干decoder layer
        self.layers = paddle.nn.LayerList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        解码器的前向传播
        :param dec_inputs:  [batch_size, tgt_len]
        :param enc_inputs: [batch_size, src_len]
        :param enc_outputs: [batch_size, src_len, d_model]
        :return:
        """
        # tgt word embedding
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]

        # 加上位置编码
        dec_outputs = self.pos_emb(dec_outputs.transpose([ 1,0,2])).transpose([ 1,0,2])  # [batch_size, tgt_len, d_model]

        # 1.生成tgt关于tgt的pad掩码，pad的为1否则为0
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]

        # 2.生成tgt语言的SubSequence掩码，每个句子是个上三角矩阵，被遮掩的是1否则为0
        dec_self_attn_subsequence_mask = get_attn_subsequent_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]

        # 将1和2处的掩码结合，这是decoder 的 self-attention需要的掩码（位置3），两个掩码在同一个位置上至少有1个是1，则取1；否则0
        dec_self_attn_mask = (dec_self_attn_pad_mask.logical_or( dec_self_attn_subsequence_mask)) # [batch_size, tgt_len, tgt_len]

        # 是decoder的 cross-attention 需要的掩码，tgt关于src的pad mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batch_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(paddle.nn.Layer):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config.src_vocab_size)  # 编码层的实例，传入源语言词典的大小
        self.decoder = Decoder(config.tgt_vocab_size)  # 解码层的实例，传入目标语言词典的大小
        # 输出层，d_model是每一个token的维度，之后会做一个到tgt_vocab_size大小的映射，然后再softmax，它们可以被解释为预测的词汇分布
        self.projection = paddle.nn.Linear(d_model, config.tgt_vocab_size, bias_attr=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        transformer的一次前向传播。
        :param enc_inputs: [batch_size, src_len]，输入的每个句子是一个样本，每个样本是个向量，向量元素是句子对应位置词语的索引
        :param dec_inputs: [batch_size, tgt_len]
        :return: dec_logits：[batch_size * tgt_len, tgt_vocab_size]，
        可以理解为，一批句子，这批句子有 batch_size*tgt_len 个单词，每个单词有 tgt_vocab_size 种情况，取其中概率最大者。
        """
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # 先编码，把编码器的最终输出送入解码器

        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # 使用编码器的信息，完成解码器的输出

        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # 将解码器的输出转换为最终的词汇空间。这个线性层通常用于生成每个目标词汇的 logits，它们可以被解释为预测的词汇分布。
        dec_logits = self.projection(dec_outputs)

        # 返回值包括重新形状的 dec_logits（以便用于交叉熵损失的计算），以及所有注意力权重。这样的输出格式有助于后续的损失计算和模型评估。
        # [batch_size * tgt_len, tgt_vocab_size]，可以理解为，一个句子，这个句子有 batch_size*tgt_len 个单词，每个单词有 tgt_vocab_size 种情况，取概率最大者
        return dec_logits.reshape(shape=[-1, dec_logits.shape[-1]]), enc_self_attns, dec_self_attns, dec_enc_attns
