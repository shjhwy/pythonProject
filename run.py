import train_model
import data_set_machine_translate

if __name__ == '__main__':
    # 训练transformer
    train_model.trainTransformer(10,data_set_machine_translate.data_loader())
    # 测试transformer