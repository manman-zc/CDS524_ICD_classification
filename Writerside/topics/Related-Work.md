# 2. Related Work

In recent years, research on automating ICD diagnostic coding has gradually emerged. Traditional methods primarily rely on feature engineering and statistical models, yet they often struggle to capture the complex semantics embedded in large-scale and diverse electronic medical record texts. With the advancement of deep learning techniques, neural network-based text classification methods (e.g., RNN, CNN) have provided novel approaches for automated coding.

The introduction of Transformer and its derivative models, such as BERT, has revolutionized natural language processing. By leveraging pre-training and fine-tuning strategies, BERT effectively learns contextual information and has demonstrated outstanding performance across various classification tasks [1]. In the medical domain, numerous studies have applied BERT to ICD coding, achieving high accuracy [2]. Particularly for Chinese electronic medical records, where both linguistic and domain-specific challenges prevail, ongoing research continues to explore and refine these approaches.

Moreover, comparative studies between traditional machine learning methods and pre-trained model-based techniques have confirmed the significant advantages of the latter in handling the complexity of medical texts [3]. These investigations provide essential theoretical and empirical support for the automated coding and diagnostic decision-making in medical contexts.

### References
[1] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  
[2] Li, X., et al. (2020). Fine-tuning Pre-trained Language Models for ICD Coding: An Empirical Study. Journal of Medical Systems, 44(3).  
[3] Wang, Y., et al. (2020). Comparison of Deep Learning Methods for Automatic ICD Coding from Electronic Medical Records. IEEE Access, 8.