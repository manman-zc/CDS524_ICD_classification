# 5. Results

本节将详细呈现本研究的实验结果，并对结果进行全面分析与讨论。我们首先展示两种方法的总体表现，然后深入分析预训练BERT模型的训练过程与性能特点。

## 5.1 总体实验结果

表5.1展示了两种方法在中文电子病历ICD诊断编码任务上的分类准确率表现。从结果可以清晰地看出，基于预训练BERT的深度学习方法明显优于传统的SVM方法，分类准确率提高了12.25个百分点。

**表5.1 不同方法的实验结果比较**

| 方法 | 分类准确率 |
|------|------------|
| SVM  | 59.00%     |
| 预训练BERT | 71.25% |

SVM方法的表现相对较弱，这可能是因为：
1. TF-IDF特征无法有效捕捉文本的语义信息和上下文关系；
2. 医疗文本中的专业术语和复杂表达使得简单的词袋模型难以准确表示；
3. 线性SVM在处理多分类问题时的固有局限性。

相比之下，预训练BERT方法显著提升了分类性能，这主要得益于：
1. BERT模型通过预训练获得了丰富的语言知识；
2. 双向注意力机制能够更好地理解医疗文本的上下文信息；
3. 共享与特定任务参数相结合的架构设计有效处理了单标签与多标签分类任务的差异；
4. 针对数据不平衡问题采取的特殊损失函数设计。

## 5.2 预训练BERT模型的训练分析

为了深入理解预训练BERT模型的训练过程和性能演变，我们对训练过程中的关键指标进行了可视化分析。

### 5.2.1 损失函数变化分析

图5.1展示了训练集与验证集在训练过程中的损失值变化趋势。

![loss_curve.png](loss_curve.png)

**图5.1 训练集与验证集的损失值变化曲线**

从损失曲线可以观察到以下几点关键发现：

1. **收敛速度**：模型在前几个epoch中快速收敛，损失值迅速下降。
2. **训练稳定性**：在第5个epoch后，训练损失继续平稳下降，而验证损失则开始呈现波动，但整体仍保持下降趋势。
3. **过拟合时机**：约在第16个epoch时，验证损失开始出现上升趋势，这表明模型可能开始过拟合训练数据。

因此，选择16个epoch作为训练轮次是合理的，既能充分利用训练数据（每轮处理45条数据），又能在过拟合出现前及时停止训练。这种训练策略平衡了模型的学习能力与泛化能力。

### 5.2.2 分类准确率分析

图5.2展示了训练集与验证集在训练过程中的分类准确率变化。

![accuracy_curve.png](accuracy_curve.png)

**图5.2 训练集与验证集的分类准确率变化曲线**

从准确率曲线可以得出以下结论：

1. **学习效果**：随着训练的进行，训练集和验证集的准确率均呈上升趋势，证明模型有效地从数据中学习了诊断编码的模式。
2. **峰值性能**：验证集准确率在第9个epoch达到峰值，此后虽有波动但整体趋于平稳，未见显著提升。
3. **模型选择**：基于"早停"(early stopping)策略[1]，我们保存了验证集准确率最高的模型参数（第9个epoch）作为最终模型，这有助于提高模型在未见数据上的泛化能力。

训练集准确率持续上升而验证集准确率在中期达到高点后趋于稳定，这种模式符合典型的深度学习模型训练特征，表明模型已经充分学习了训练数据中的规律，并且具有良好的泛化能力。

### 5.2.3 不确定性权重参数分析

图5.3展示了不确定性加权损失函数中sigma参数在训练过程中的变化趋势。

![sigma_curve.png](sigma_curve.png)

**图5.3 不确定性加权损失中sigma值的变化曲线**

从sigma值的变化可以观察到：

1. **参数稳定性**：两个sigma值（$\sigma_1$和$\sigma_2$）在训练过程中仅在小范围内波动，表明初始设置（$\sigma_1=-1$，$\sigma_2=1$）已经相对合理，能够有效平衡两个任务的损失。
2. **任务平衡**：sigma值的相对稳定证明了不确定性加权损失函数成功地动态平衡了单标签分类任务和多标签分类任务，避免了任一任务主导训练过程的情况。
3. **收敛特性**：两个sigma值在训练后期趋于稳定，表明模型已找到两个任务间的最优权重分配。

这种稳定的sigma值变化表明我们的多任务学习框架成功地在保持两个任务相对独立性的同时，又实现了它们之间的协同优化，这与Kendall等人[2]提出的多任务学习理论一致。


## 参考文献

[1] Prechelt, L. (1998). Early stopping-but when?. In Neural Networks: Tricks of the trade (pp. 55-69). Springer, Berlin, Heidelberg.

[2] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).

[3] Chung, J. W., Yang, J., & Yahyavi, M. (2022). Clinical text classification with rule-based features and knowledge-guided deep learning. BMC medical informatics and decision making, 22(1), 1-15.

[4] Mullenbach, J., Wiegreffe, S., Duke, J., Sun, J., & Eisenstein, J. (2018). Explainable prediction of medical codes from clinical text. arXiv preprint arXiv:1802.05695.
