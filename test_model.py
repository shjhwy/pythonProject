import paddle
from data_set_machine_translate import tgt_vocab
from data_set_machine_translate import idx2word
from model.transformer_mine import Transformer


def greedy_decoder(model: Transformer, enc_input: paddle.Tensor, start_symbol: str):
    """
    我们不知道目标序列输入。因此，我们尝试逐字生成目标输入，然后将其输入到模型中。
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target
    """
    # 单独使用模型的编码器处理 enc_input，获取Encoder输出的 enc_outputs 和自注意力权重 enc_self_attns。
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 解码器的输入序列，这个张量在第一个维度中有一个元素（表示一个句子），但是这个元素本身是一个空的张量，即没有任何数据
    dec_input = paddle.zeros(shape=[1, 0], dtype=enc_input.dtype)

    terminal = False
    next_symbol = start_symbol

    # 开始循环，直到遇到终止条件（例如遇到句号或达到最大长度）。
    while not terminal:
        # 把将上一次新生成的词添加到已存在的解码器输入中
        dec_input = paddle.concat([dec_input.detach(), paddle.to_tensor([[next_symbol]], dtype=enc_input.dtype)], -1)

        # 解码 [1, tgt_len, d_model]
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)

        # 手动映射成 [1,tgt_len, tgt_vocab_size]
        projected = model.projection(dec_outputs)

        # squeeze(0) 方法用于移除张量中大小为 1 的维度，这里指的是批次维度。移除后，张量的形状变为 [tgt_len, tgt_vocab_size]。
        # 这里 dim=-1 指定了在最后一个维度（词汇维度）上寻找最大值，即寻找每个位置最可能的词汇。max() 返回两个值：最大值和对应的索引。通过添加 [1]，我们只取最大值的索引，这些索引对应于预测词汇的 ID。
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]

        # -1，是取的当前解码器输出的序列的最后一个词，也就是最新生成的这个词
        next_word = prob.data[-1]

        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            # 终止词
            terminal = True
        print(next_word)

        # 输出预测的完整句子
    return dec_input


def test_transformer(test_data: paddle.io.DataLoader, model: Transformer):
    model.eval()
    enc_inputs, _, _ = next(iter(test_data))
    for i in range(len(enc_inputs)):
        # 通过贪婪算法，逐步获得解码器的完整输入
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
        # 通过模型预测最终的词汇索引
        predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
        # 将预测的索引转换为实际的词汇，并打印出来。
        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
