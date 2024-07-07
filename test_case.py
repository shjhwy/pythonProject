import unittest
import data_set_machine_translate
import data_set_mnist
import train_model

class MyTestCase(unittest.TestCase):
    def test_transformer(self):
        # 训练transformer
        train_model.train_test_transformer(2, data_set_machine_translate.data_loader())
        # 测试transformer
        self.assertEqual(True, True)  # add assertion here

    def test_lenet(self):
        train_data, test_data = data_set_mnist.dataLoader()
        # 训练lenet
        train_model.train_lenet(2, train_data)

        self.assertEqual(True, True)  # add assertion here

if __name__ == '__main__':
    unittest.main()
