# Chinese electronic medical record ICD diagnosis code classification based on fine-tuning pre-trained BERT

## Abstract
This study proposes an automatic classification system for Chinese electronic medical record ICD diagnosis codes based 
on the pre-trained BERT model. With the rapid growth of electronic medical record data, an efficient and accurate 
automatic coding system has become increasingly important for medical institution management and clinical decision support. 
Considering the characteristics of Chinese medical texts and the complexity of ICD coding, we designed a method that 
integrates entity extraction and multi-task learning. First, we used PaddleNLP to perform medical entity extraction and 
relationship extraction on medical record texts to extract key information from unstructured texts. Then, based on the BERT model, 
we constructed a dual-task learning architecture with the first 6 layers sharing parameters and the last 6 layers being task-specific, 
simultaneously handling single-label classification of the main diagnosis and multi-label classification of other diagnoses. 
To address the challenge of data imbalance, we introduced class weight adjustment and an uncertainty-weighted loss function, 
effectively improving the recognition ability of rare categories. Experimental results show that our method achieved a 
classification accuracy of 71.25% on the 2025 Tianchi competition Chinese electronic medical record ICD diagnosis 
coding evaluation dataset, significantly better than the traditional SVM method's 59.00%. In addition, we developed a 
user-friendly interactive system interface that supports real-time prediction, providing convenient assistance for 
clinical coding work. This study not only verifies the advantages of deep learning in medical text coding but also 
provides practical methods for solving similar medical natural language processing tasks.

**Keywords**: electronic medical record; ICD diagnosis coding; BERT; multi-task learning; uncertainty-weighted loss; medical text processing; deep learning