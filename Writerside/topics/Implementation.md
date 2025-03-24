# 4. Implementation

本节将详细介绍两种核心实现方法——传统机器学习方法(SVM)和深度学习方法(BERT)的具体实现细节，包括模型架构、实验环境配置以及训练过程中遇到的挑战与解决方案。

## 4.1 模型实现

### 4.1.1 SVM方法实现

支持向量机(SVM)作为一种经典且高效的机器学习算法，在文本分类任务中表现出色[1]。本研究中，SVM实现的主要步骤如下：

1. **特征提取**：采用TF-IDF (Term Frequency-Inverse Document Frequency)方法提取文本特征。选择了5000个最具代表性的特征，将非结构化的病历文本转换为机器学习算法可处理的向量表示。
   
2. **数据划分**：数据集包含800条训练数据和200条测试数据，保持了原始数据集的分布。

3. **模型训练**：基于sklearn库实现SVM模型，使用线性核函数以平衡计算效率与分类性能。

SVM方法由于其计算效率高、对小样本学习能力强的特点，作为本研究的基线模型，为评估更复杂模型的性能提供了参考标准。

### 4.1.2 预训练BERT方法实现

BERT (Bidirectional Encoder Representations from Transformers)作为近年来自然语言处理领域的突破性技术[2]，其双向编码机制能够更全面地捕获文本的语义信息。本研究基于中文预训练模型bert-base-chinese进行了任务特定的微调，具体实现如下：

1. **模型架构设计**：采用了共享参数与任务特定参数相结合的架构设计。如图所示，BERT模型的前6层参数在两个分类任务间共享，这基于的假设是底层特征提取对不同的分类任务具有通用性；而后6层则针对两个不同的分类任务（主诊断单标签分类和其他诊断多标签分类）分别训练独立的参数，以捕获任务特定的特征。

![BERT结构说明.png](BERT结构说明.png)

2. **数据划分**：训练集720条，验证集80条（按9:1比例划分），测试集200条。这种划分方式确保了模型在训练过程中能够不断评估其泛化能力。

## 4.2 实验环境配置

为确保研究的可复现性，详细记录了实验的软硬件环境：

- **计算平台**：Google Colab Pro
- **计算资源**：NVIDIA T4 GPU (15GB显存)
- **数据预处理框架**：
  - paddlepaddle-gpu 2.6.0
  - paddlenlp 2.8.1
- **模型实现框架**：
  - SVM：scikit-learn
  - BERT：PyTorch和Transformers库

## 4.3 实施挑战与解决方案

在实验过程中，我们面临的主要挑战是诊断编码的分布不均衡问题，特别是其他诊断编码的分布极度不平衡。这种数据不平衡会导致模型偏向于学习主要诊断编码，而忽略其他诊断编码中的少数类别。为解决此问题，我们采用了两种互补的方法：

### 4.3.1 类别权重调整

为平衡不同类别的影响，我们引入了类别权重机制。对于多标签分类任务（其他诊断编码），我们根据各类别在训练集中的出现频率计算反比例权重，范围在1至500之间。这使得模型在训练过程中对稀有类别给予更高的关注度，如下图所示：

![pos_weights.png](pos_weights.png)

- 对于每个类别：
  - `total_samples - label2_counts` 是负样本数（不属于该类别的样本数）。
  - `label2_counts` 是正样本数。
  - `1e-6` 是一个小值（ε），用于防止除以零（当某个类别没有正样本时）。

公式如下：

$$
\text{pos weights}[j] = \frac{\text{total samples} - \text{label2 counts}[j]}{\text{label2 counts}[j] + 10^{-6}}
$$


这种方法有效提高了对低频类别的识别能力，从而改善了整体的分类性能。

### 4.3.2 不确定性加权损失函数

为处理单标签和多标签分类任务之间的平衡问题，我们实现了Kendall等人[4]提出的不确定性加权损失函数(Uncertainty to Weigh Losses)。这种方法允许模型自动学习每个任务的最优权重，而不需要手动调整。

具体实现如下：

1. **单标签分类任务的损失函数**（交叉熵损失）：

![单分类任务损失函数.png](单分类任务损失函数.png)

$$
L_1 = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})
$$

#### 参数说明：
- **$ N $**：样本总数。
- **$ C $**：类别总数。
- **$ y_{i,c} $**：样本 $i$ 在类别 $c$ 的真实标签（通常以 one-hot 形式表示）。
- **$ \hat{p}_{i,c} $**：样本 $i$ 在类别 $c$ 的预测概率，计算方式为：
  $$
  \hat{p}_{i,c} = \text{softmax}(\text{logits1})_{i,c}
  $$

#### 备注：
`softmax` 函数将 logits 转换为概率分布，确保 **$ \hat{p}_{i,c} $** 的值在 $[0, 1]$ 范围内。


2. **多标签分类任务的损失函数**（带权重的二元交叉熵损失）：

![多分类任务损失函数.png](多分类任务损失函数.png)

$$
L_2 = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_p \cdot y_i \cdot \log(\hat{p}_i) + (1 - y_i) \cdot \log(1 - \hat{p}_i) \right]
$$

#### 参数说明：
- **$N$**: 样本数量。
- **$y_i$**: 第 $i$ 个样本的真实标签（0 或 1）。
- **$\hat{p}_i$**: 预测概率，计算方法为：
  $$
  \hat{p}_i = \text{sigmoid}(\text{logits2}_i)
  $$
- **$w_p$**: 正类权重，用于调整正负样本的不平衡。

#### 备注：
这个公式适用于不平衡数据集中的二分类问题，通过权重 $w_p$ 对正负样本的重要性进行调整。


3. **综合损失函数**：

![总损失函数.png](总损失函数.png)

$$
L_{total} = \frac{L_1}{2\sigma_1^2} + \frac{L_2}{2\sigma_2^2} + \log(\sigma_1^2) + \log(\sigma_2^2)
$$

#### 参数说明：
- $L_1$ 和 $L_2$ 表示两个部分的损失函数。
- $\sigma_1^2$ 和 $\sigma_2^2$ 分别表示两个损失部分的方差。
- $\log(\sigma_1^2)$ 和 $\log(\sigma_2^2)$ 是对方差的对数项，用于正则化或处理方差。


其中，我们设置单标签分类的$\sigma_1$为-1，多标签分类的$\sigma_2$为1，作为初始值。在训练过程中，这些参数会随着梯度下降自动调整，实现任务间的动态平衡。

这种不确定性加权损失函数的引入显著改善了两个任务的联合优化效果，提高了模型在两种分类任务上的整体性能。

## 参考文献

[1] Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In European conference on machine learning (pp. 137-142). Springer, Berlin, Heidelberg.

[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[3] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

[4] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).

