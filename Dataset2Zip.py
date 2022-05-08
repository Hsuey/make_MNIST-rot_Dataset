datasetType = "FashionMNIST" #  FashionMNIST or MNIST
saveDir = ""
# datasetType = 'w'

import os
import zipfile
 
 
def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()

if datasetType == "FashionMNIST":
        saveDir = "fashionmnist_rotation_new"
else:
    saveDir = "mnist_rotation_new"
zipDir(saveDir, saveDir + ".zip")
print(f"zip OK. loacate in {saveDir}.zip")