# 基于 U-Net 结构的可逆图像隐写方案（PyTorch 非官方实现）

本项目为以下论文的 PyTorch 实现，供学习与研究参考：

> **《Reversible Image Steganography Scheme Based on a U-Net Structure》**  
> 作者：Xintao Duan 等  
> 发表：IEEE Access, 2019  
> [DOI 链接](https://doi.org/10.1109/ACCESS.2019.2891247)

---

## 📌 项目简介

该方法基于深度神经网络，通过 U-Net 结构将一张秘密图像嵌入另一张载体图像中，生成高保真、可逆的隐写图像。网络包含两个模块：

- 🧠 隐写网络（Hiding Net）：U-Net 编码器结构
- 🔓 提取网络（Extraction Net）：浅层卷积网络

主要特点：

- 全图像端到端输入输出
- 可逆恢复原始秘密图像
- 高 PSNR/SSIM，低可检测性

---

## 📁 项目结构

```text
.
├── HidingNet.py          # 隐写网络（U-Net 结构）
├── Extractor.py          # 提取网络
├── main.py               # 训练主程序
└── README.md             # 当前说明文档
````

---

## 🔧 网络结构

### 隐写网络（Hiding Network）

* 输入：拼接后的 6 通道图像（Cover + Secret）
* 架构：7 层下采样 + 7 层上采样（U-Net）
* 输出：3 通道隐写图像（Stego）

### 提取网络（Extraction Network）

* 输入：Stego 图像
* 架构：6 层卷积网络（无池化）
* 输出：恢复后的 Cover 和 Secret 图像

---

## 🧠 损失函数

训练过程中使用如下损失函数：

```
L = ‖c - c′‖ + α × ‖s - s′‖
```

* `c`：原始载体图像，`c′`：重建的载体图像
* `s`：原始秘密图像，`s′`：提取出的秘密图像
* `α`：平衡系数，默认设为 0.75

---

## 📊 评价指标

| 指标   | 说明             |
| ---- | -------------- |
| PSNR | 峰值信噪比，衡量像素误差   |
| SSIM | 结构相似性指数，衡量图像结构 |

---

## 📈 指标示例（ImageNet 测试）

| 比较对象                | PSNR (↑) | SSIM (↑) |
| ------------------- | -------- | -------- |
| Cover vs Stego      | 40.47    | 0.9794   |
| Secret vs Extracted | 40.66    | 0.9842   |

---

## 📚 引用原始论文

如果你使用了本项目，请引用原始论文：

```bibtex
@article{duan2019reversible,
  title={Reversible image steganography scheme based on a U-Net structure},
  author={Duan, Xintao and Jia, Kai and Li, Baoxia and Guo, Daidou and Zhang, En and Qin, Chuan},
  journal={Ieee Access},
  volume={7},
  pages={9314--9323},
  year={2019},
  publisher={IEEE}
}
```

---

## ⚠️ 声明

本项目为论文的**非官方实现**，仅用于学术研究与教学用途，与原作者无直接关系。

---

## 📬 联系方式

如有问题欢迎提交 Issue，或通过邮箱联系我：**[abracasora@gmail.com](abracasora@gmail.com)**
