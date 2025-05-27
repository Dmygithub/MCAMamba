
# 🚀 **MCAMamba: 多源遥感图像分类的多尺度特征融合状态空间模型**

[![GitHub](https://img.shields.io/badge/GitHub-MCAMamba-green)](https://github.com/yourusername/MCAMamba)

---

## 📌 **简介**

MCAMamba是一个专为多源遥感图像分类设计的多尺度特征融合模型。通过利用状态空间建模技术，MCAMamba有效捕获空间和光谱依赖关系，确保高精度和计算效率。

### 🔍 **主要特点**

✅ 多尺度特征提取  
✅ 跨模态数据融合  
✅ 高效的状态空间表示  
✅ 增强型多源遥感数据融合  

---

## 📂 **数据集**  

我们实验中使用的数据集可从以下链接获取：  
📥 **[下载柏林和奥格斯堡数据集](https://github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS)** 

---

## 🛠 **安装与依赖**

运行代码前，请确保安装以下依赖：

```bash
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.0.1
```

---

## 🏋️‍♂️ **使用方法：训练MCAMamba**

要在柏林数据集上训练MCAMamba，使用以下命令：

```bash
python train.py --epoch 40 --lr 1e-4 --batchsize 128 --dataset Berlin
```

### 🔧 **训练参数**:

- `--epoch`: 训练轮数
- `--lr`: 学习率
- `--batchsize`: 批处理大小
- `--dataset`: 数据集名称

---

## 📬 **联系方式**

如有任何问题，请通过电子邮件联系我们：  
📧 [youremail@example.com](mailto:youremail@example.com)

---

## 📚 **引用**

如果您在研究中使用了MCAMamba，请引用我们的工作：

```
@article{mcamamba2023,
  title={MCAMamba: 多源遥感图像分类的多尺度特征融合状态空间模型},
  author={您的姓名},
  year={2023}
}
```
