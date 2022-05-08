# FashionMNIST-rot and MNIST-rot
这是生成FashionMNIST-rot 和 MNIST-rot的代码。
## 如何运行：
### 运行方法一：
- 对这三个文件（`download_mnist.py`、`make_mnist-rot.py`、`Dataset2Zip.py`）的第一行设置：`datasetType = "FashionMNIST" #  FashionMNIST or MNIST`, 选择生成哪一个数据FashionMNIST-rot 还是 MNIST-rot
- 然后依次运行 `download_mnist.py`、`make_mnist-rot.py`、`Dataset2Zip.py`
### 运行方法二：
- 直接运行 make_mnist_rot_dataset.ipynb

## 生成的dataset格式如下：
> 训练集 + 验证集：12000，测试集：50000
- mnist_rotation_new.zip
	- mnist_all_rotation_normalized_float_test.amat
	- mnist_all_rotation_normalized_float_train_valid.amat
