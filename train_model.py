import model.transformer as transformer
import data_set_machine_translate as transformer_data
import paddle


def trainTransformer(epochs:int,data_loader:paddle.io.DataLoader):
    """
    训练 transformer
    :param epochs: 迭代次数
    :param data_loader: 数据集Loader，需要继承paddle.io.DataLoader类型
    :return:无
    """
    # mac没有gpu，所以注意不能在代码中调用显式地把数据放在gpu上的操作，比如调用.cuda()
    paddle.set_device('cpu')
    config=transformer.Config(transformer_data.src_vocab_size, transformer_data.tgt_vocab_size)
    model = transformer.Transformer(config)  # 调用Transformer模型
    # 设置为训练模式，允许 Dropout
    model.train()
    # 在计算损失时忽略特定的类别标签。在这里，ignore_index=0 指定标签为 0 的目标数据不应该对损失计算产生贡献。
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=0)  # 交叉熵损失函数
    # 此处必须要显式地给出参数parameters
    optimizer = paddle.optimizer.SGD(parameters=model.parameters())

    # 开始迭代epoch
    for epoch in range(epochs):
        # 遍历每个batch
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """

            # 走一遍transformer的前向传播
            # 输出outputs: [batch_size * tgt_len, tgt_vocab_size]，表示在batch_size * tgt_len位置上，每个可能的词的概率。
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

            # dec_outputs是当前批数据每个样本的标签，把它展平称一维，这是计算交叉熵损失时要求的标签的形状
            # dec_outputs.view(-1):[batch_size * tgt_len]
            # 所以，虽然一个样本是一行句子，但是对于transformer，计算损失是对句子里的每个单词计算损失的，而不是一个样本
            # view的写法与torch写法不一致，形状必须传入一个tuple|list
            loss = criterion(outputs, dec_outputs.reshape(shape=[-1]))
            print('Epoch:', '%02d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.numpy()))

            optimizer.clear_grad()  # 在反向传播之前，清除（归零）之前的梯度信息
            loss.backward()  # 对损失进行反向传播计算，自动计算模型参数的梯度。
            optimizer.step()  # 使用优化器更新模型的权重，以最小化损失函数。