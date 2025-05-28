#  **MCAMamba: Multi-scale Feature Fusion State Space Model for Multi-source Remote Sensing Image Classification**

[![GitHub](https://img.shields.io/badge/GitHub-MCAMamba-green)](https://github.com/yourusername/MCAMamba)

---

## 📌 **Introduction**

MCAMamba is a multi-scale feature fusion model designed specifically for multi-source remote sensing image classification. By utilizing state space modeling techniques, MCAMamba effectively captures spatial and spectral dependencies, ensuring high accuracy and computational efficiency.


## 📂 **Dataset**  

The datasets used in our experiments can be obtained from the following link:  
📥 **[Download Houston2013 Dataset](https://pan.baidu.com/s/12-hGPcoTseVdUEO_1Ypp1w?pwd=xszv)** 

---

## 🛠 **Installation and Dependencies**

Before running the code, please ensure the following dependencies are installed:

```bash
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.0.1
```

---

## 🏋️‍♂️ **Usage: Training MCAMamba**

To train MCAMamba on the Houston2013 dataset, use the following command:

```bash
python train.py --epoch 40 --lr 1e-4 --batchsize 128 --dataset Houston2013
```


## 📬 **Contact**

For any questions, please contact us via email:  
📧 [doumingyu24@mails.ucas.ac.cn](doumingyu24@mails.ucas.ac.cn)

---

## 📚 **Citation**

If you use MCAMamba in your research, please cite our work:

```
@article{mcamamba2025,
  title={MCAMamba: Multi-scale Feature Fusion State Space Model for Multi-source Remote Sensing Image Classification},
  author={Mingyu Dou},
  year={2025}
}
```
