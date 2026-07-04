Classification of Organic and Recyclable Waste
based on Feature Extraction and Machine Learning
Algorithms
Elham Tahsin YASINa, Murat KOKLUb
a Graduate School of Natural and Applied Sciences
Selcuk University, Konya, TÜRKİYE
ilham.tahsen@gmail.com,
b Department of Computer Engineering
Selcuk University, Konya, TÜRKİYE
mkoklu@selcuk.edu.tr
Abstract— Managing solid waste effectively requires the proper
classification of waste. To determine whether a waste is organic or
recyclable, machine learning methods can be used. This study
extracted features from waste samples using the InceptionV
feature extraction method, and three machine learning classifiers
were used to compare accuracy. The study utilized the
InceptionV3 deep convolutional neural network, which was pre-
trained on large-scale image datasets and fine-tuned on waste
images to extract features. The extracted features were used to
train three machine-learning classifiers. The performance of the
classifiers was evaluated using a labeled waste image dataset. As a
result of our experiments, we found that SVMs achieved an
accuracy of 96.3% without any feature selection, Decision Trees
achieved a result of 85.8%, and KNNs achieved a result of 94.9%.
Based on our study, we demonstrate that it is feasible to classify
solid waste using machine learning algorithms. A waste
classification and management system that achieves optimum
efficiency can be implemented with the help of the findings of this
study.
Keywords— Solid waste classification, Feature extraction, Waste
management, Machine learning algorithms, Deep learning-based
approach.
I. INTRODUCTION
Solid waste management and disposal have become
significant challenges worldwide due to the increasing
generation of waste. Proper waste management systems are
crucial for promoting a sustainable future and mitigating the
environmental impact of waste accumulation [1]. Classifying
waste materials into organic and recyclable categories can
significantly contribute to efficient recycling processes and
reducing waste in landfills. However, effective waste
classification is often hampered by the complex nature of solid
waste materials and the logistical and economic constraints
associated with waste management [2]. The incorporation of
modern technologies such as machine learning, computer
vision, and sensor-based systems can simplify waste
classification procedures and boost the effectiveness of waste
management systems [3]. This study focuses on the application
of machine learning techniques for the classification of solid
waste materials, with a specific focus on organic and recyclable
waste. The primary objective is to design and implement a
model capable of accurately analysing image datasets to
identify organic and recyclable solid waste materials. The study
employs multiple machine learning algorithms such as Support
Vector Machine, K-Nearest Neighbours, and Decision Trees to
classify waste materials into organic and recyclable categories.
Using machine learning algorithms to perform statistical
analyses is the primary objective of this study to develop an
automated system capable of accurately classifying solid waste
materials into organic and recyclable categories.
In recent years, the efficient management and disposal of
solid waste have become a significant concern for
municipalities and environmentalists worldwide. The
increasing generation of waste and the subsequent challenges
associated with its proper disposal have become pressing issues
worldwide [4]. For a sustainable future and to mitigate the
environmental impact of waste accumulation, effective waste
management systems are crucial. Among the various
approaches to waste management, the separation and
classification of waste materials into organic and recyclable
categories can significantly contribute to efficient recycling
processes and the reduction of waste in landfills.
Organic waste comes from plants or animals and can be
decomposed by microorganisms. Examples include food scraps,
yard trimmings, and agricultural residues. Recyclable waste
can be reprocessed into new products, reducing the demand for
raw materials and conserving energy. Common recyclable
materials include paper, cardboard, plastics, metals, and glass
[5].
The proper classification of solid waste into organic and
recyclable categories is important for waste management [6].
Organic waste can be processed to produce compost or biogas,
while recyclable waste can be reintroduced into the production
cycle to reduce the environmental impact of consumption and
manufacturing [7].
The diverse nature of solid waste materials and logistical
constraints associated with waste management have led to
growing interest in developing innovative approaches and
technologies to improve waste classification processes [2].
Researchers and practitioners have been exploring advanced
technologies like computer vision, machine learning, and
sensor-based systems to develop automated systems that can
efficiently classify waste materials into organic and recyclable
categories. These technologies have the potential to contribute
to sustainable waste management practices and promote a
cleaner and healthier environment [8].
This study highlights the importance of efficient
classification of solid waste, especially into organic and
recyclable categories, for sustainable waste management
practices. It discusses how traditional manual sorting methods
are labor-intensive and prone to human error and how machine
learning techniques can be used to enhance waste classification.
Organic waste includes biodegradable materials from plants
and animals, while recyclable waste can be reprocessed into
new products, reducing the need for raw materials [1].
The research aims to develop a machine learning model that
can accurately classify solid waste materials into organic and
recyclable categories using image datasets. The study uses the
InceptionV3 deep learning-based image recognition model to
extract features from the images and provide a strong
foundation for classification. The goal is to automate the waste
classification process and improve the efficiency of waste
management practices [9, 10].
The study uses machine learning algorithms such as SVM,
Decision Trees, and KNN to classify waste materials into
organic and recyclable categories. The feature extraction stage
uses InceptionV3, a deep learning-based image recognition
model, to extract salient features from the image dataset [11].
The study systematically decreases the number of features to
find the optimal balance between computational efficiency and
classification accuracy [9].
This study aims to develop an automated system for
classifying solid waste materials into organic and recyclable
categories using machine learning algorithms such as SVM,
Decision Trees, and KNN in combination with InceptionV
deep learning model for feature extraction. The goal is to
contribute to the development of efficient and sustainable waste
management practices.
II. RELATED WORKS
Zhang et al., have contributed to the field of waste image
classification can be greatly improved with the use of advanced
technologies such as deep learning, computer vision, and
sensor-based systems. By developing a waste image
classification model called CTR, which is based on the
ResNet18 architecture and includes a self-monitoring module
to enhance feature map representation and capture unique waste
image features. Their model achieved a high classification
accuracy of 95.87% when tested on the TrashNet dataset.
However, due to the limited size of this dataset, the CTR
model's ability to classify real-world waste materials may be
limited [5].
Uganya et al., introduced an IoT-based approach for smart
waste management that employs machine learning
classification techniques to forecast waste possibility. Their
method monitors the wastage capacity, gas level, and metal
level using IoT-enabled dustbins. Testing of the proposed
method showed that it achieved an accuracy of 92.15% with a
random forest algorithm and had a low time consumption of 0.
milliseconds, which outperforms other classification
techniques [12].
Girsang et al., proposed a study that compares VGG16 and
Inception V3 CNN algorithms for image recognition in waste
management and proposes algorithms using semi-supervised
learning to train from different images. VGG16 outperformed
Inception V3 for the given dataset, highlighting the need for
model optimization in waste management applications. Waste
classification can be achieved with deep learning algorithms,
and high accuracy of 91.60% and 91.25% was achieved for
VGG16 and Inception V3 respectively. Further research is
needed to improve both models for better accuracy [13].
Altikat et al., discussed the classification of paper, glass,
plastic, and organic waste images using deep convolutional
neural networks. The study compares the performance of four-
layer and five-layer architectures and finds that the five-layer
architecture achieved a 70% accuracy rate in waste
classification, while the four-layer architecture achieved a
61.67% accuracy rate. The accuracy rate for differentiating
plastic waste was lower than other types of waste. The accuracy
rate for organic waste classification was higher than other types
of waste. The study suggests that machine learning techniques
can minimize the human factor and improve the performance
of recycling facilities [14].
Nnamoko et al., focused on the development of a bespoke
five-layer convolutional neural network to improve waste
classification accuracy from images. The dataset contains
25,077 images categorized into organic and recyclable waste.
The study evaluates the performance of the model using two
different image resolutions to investigate the trade-off between
computational cost and accuracy. The results indicate that a
smaller image resolution leads to a lighter model with less
training time and better accuracy compared to the larger model.
The study also highlights the importance of methodology
transparency and reproducibility of results, and all
experimental artifacts are made available in a public repository
[15].
Adedeji and Wang, proposed a machine learning-based
system for intelligent waste material classification, which
simplifies the waste separation process by categorizing it into
different components, such as plastic, paper, metal, and glass.
To achieve this, they utilized a pre-trained ResNet- 50
Convolutional Neural Network model as an extractor and a
Support Vector Machine (SVM) for classification. The system
yielded an accuracy of 87% when tested on a dataset of trash
images, reducing the need for human intervention and speeding
up the waste separation process [16].
Cha et al., proposed a novel approach to construction and
demolition waste management using artificial intelligence (AI).
They developed and tested hybrid machine learning (ML)
models by combining three algorithms, including support
vector regression (SVR), artificial neural network (ANN), and
a combination of an autoencoder (AE) and the random forest
(RF) algorithm. Results showed that the hybrid AE-ANN
model performed better than other non-hybrid and hybrid
models, achieving up to a 49% improvement in MAE, RMSE,
R2, and R. The proposed approach can be used to develop a
demolition waste generation rate (DWGR) ML model and
improve performance in various fields [17].
III. MATERIAL AND METHODS
The study presented here, titled "Classification of Organic
and Recyclable Waste based on Feature Extraction and
Machine Learning Algorithms," aims to address this issue by
leveraging the power of computer vision and machine learning
techniques. Using an image dataset for solid waste
classification, to achieve this, the study first employs
InceptionV3, a deep learning-based image recognition model,
to extract relevant features from the image dataset. By selecting
2048 features, the model captures the essential characteristics
of the waste materials in question. Subsequently, various
machine learning algorithms, such as Support Vector Machine
(SVM), Decision Trees, and K-Nearest Neighbours (KNN), are
utilized to classify waste into organic and recyclable categories.
Fig. 1. illustrates the graphical representation of the proposed
study.
Through this innovative approach, the study contributes to
the growing body of research on waste classification,
demonstrating the potential of integrating advanced machine
learning techniques with traditional waste management
practices to promote a cleaner and more sustainable
environment.
Fig. 1 Flow Diagram of Solid Waste Classification Study
A. Waste Classification Dataset
This dataset used in the study was obtained from the
Mendeley Data website. There are 24,705 images of solid
household waste in the dataset, which have been divided into
two categories: organic (13,880 images) and recyclable (10,
images). It is a restructured version of a Kaggle dataset
consisting of 25,077 images, with some clean-up operations
performed to reduce it [18]. Fig. 2 illustrates the image samples
of the dataset.
Fig. 2 Waste Classification Dataset Sample Images [18].
B. Feature Extraction Using InceptionV
Feature extraction is a crucial step in the development of
effective machine learning models, as it involves the
identification and representation of informative characteristics
from the raw data that can be used for subsequent classification
or regression tasks [19]. Deep learning-based models [13], such
as InceptionV3, have emerged as powerful tools for feature
extraction, particularly in the domain of image recognition and
analysis.
InceptionV3 is a state-of-the-art deep learning model for
image recognition, which is based on the concept of inception
modules [20]. The InceptionV3 architecture is designed to
address the limitations of traditional convolutional neural
networks (CNNs) [13, 14], such as computational efficiency
and the ability to handle large-scale image datasets effectively.
The key innovation of the InceptionV3 architecture lies in
the inception modules, which are building blocks that enable
the model to efficiently learn complex feature representations
at multiple scales [21]. Inception modules consist of parallel
convolutional branches with different filter sizes, pooling
operations, and dimensionality reduction techniques. By
aggregating the outputs of these branches, InceptionV3 can
capture a diverse set of features and spatial information from
the input images, leading to improved recognition and
classification performance [20].
Feature extraction with InceptionV3 typically involves using
the pre-trained model as a fixed feature extractor, where the
final fully connected layers are removed, and the output of the
remaining convolutional layers is used as a feature
representation for the input images [22]. This process results in
a compact and informative feature vector, which can be used as
input for various machine learning algorithms, such as Support
Vector Machines (SVM), K-Nearest Neighbours (KNN), and
Decision Trees, to perform classification or regression tasks
[21].
C. Cross-validation
Cross-validation is a widely used technique in machine
learning for evaluating the performance and generalization
ability of models. It involves splitting the dataset into several
subsets, with one subset being used as the test set and the rest
for training [23]. This process is repeated multiple times, with
different subsets used for testing and training, to obtain an
average performance metric that is less sensitive to the specific
choice of subsets [24]. The 10-fold cross-validation technique
is a common implementation of this approach, where the
dataset is randomly divided into 10 equal parts, and the model
is trained and tested on different subsets in a repeated manner.
Cross-validation is crucial in preventing overfitting and
obtaining more robust performance estimates, which can aid in
selecting the best model for a given task [24-26].
D. Machine learning
The use of machine learning is crucial for automated image
classification because it allows for the identification and
categorization of images based on visual characteristics.
Previous methods relied on hand-crafted features and rule-
based algorithms, which were time-consuming and often
inaccurate [19]. Machine learning algorithms can recognize
patterns and features in images, making the classification
process faster and more precise. This is especially important in
fields such as healthcare, security, and environmental
monitoring, where accurate image identification is vital.
Support Vector Machines (SVM) [24], K-Nearest Neighbors
(KNN), and Decision Trees are popular machine learning
algorithms with unique strengths and weaknesses that make
them suitable for different problems and datasets. This study
explains the principles, benefits, and limitations of SVM,
Decision Trees, and KNN in the context of machine learning
applications [27].
1) SVM
Support Vector Machines (SVM) is a supervised learning
algorithm that has gained considerable popularity for
classification and regression tasks in machine learning. The
algorithm constructs a hyperplane that optimally separates the
data points into distinct classes within each feature space. The
margin, defined as the distance between the hyperplane and the
closest data points, or support vectors, is maximized by SVM.
One of the strengths of SVM is its ability to handle both linearly
and non-linearly separable data, which is achieved using
various kernel functions. This makes SVM a versatile
algorithm for various types of problems, especially those with
high-dimensional data [12, 24, 28, 29].
2) Decision Tree
Decision Trees are machine-learning models that can be
used for classification and regression tasks. They are known for
their interpretability and ease of use. The Decision Tree
algorithm partitions the input feature space into homogeneous
regions based on a set of decision rules derived from the
training data. The tree structure represents a hierarchical
organization of these rules, with internal nodes corresponding
to feature splits and leaf nodes representing the predicted class
or output value. Decision Trees are particularly useful for
problems involving categorical or mixed-type data, and their
graphical representation facilitates easy interpretation and
understanding of the underlying decision-making process [12,
26, 28, 29].
3) KNN
K-Nearest Neighbours (KNN) is a versatile, instance-based
learning algorithm used primarily for classification, but it can
also be applied to regression tasks. KNN operates by
identifying the K nearest training instances to a given query
point and predicting the class or output value based on a
majority vote or weighted average of these neighbouring
instances [30]. KNN is a non-parametric and lazy learning
algorithm, which means that it makes no assumptions about the
underlying data distribution and does not require a specific
model to be trained [31]. This flexibility allows KNN to adapt
well to various problem domains, especially when dealing with
small or noisy datasets [28].
Table I displays the default parameters for MATLAB of each
model (SVM, KNN, and Tree). The same default parameters
were used for the performance of the algorithms [32, 33].
TABLE I
MODEL'S HYPERPARAMETERS
Machine
Learning Models Model parameters^
SVM
Kernel function: Cubic
Box constrain level: 1
Kernel scale: Automatic
Standardize data: Yes
Multiclass method: One-vs-One
Decision Tree
Preset: Fine Tree
Surrogate decision split: off
Split’s criterion: Gini’s diversity index
Maximum number of splits: 100
KNN
Preset: Weighted KNN
Distance metric: Euclidean
Number of neighbors: 10
Standardize data: Yes
Distance weight: Squared inverse
While SVM, Decision Trees, and KNN have demonstrated
remarkable success in numerous applications, it is important to
recognize their potential limitations and challenges. For
instance, SVM can be sensitive to the choice of kernel function
and hyperparameters, Decision Trees may suffer from
overfitting and instability, and KNN can be computationally
expensive and sensitive to the choice of distance metric and K
value. Despite these challenges, these three algorithms remain
essential tools in the machine learning toolbox, and their
continued development and application hold great promise for
advancing our understanding and solving complex real-world
problems.
E. Confusion Matrix and Evaluation Metrics
Machine learning algorithms are assessed through statistical
metrics that allow for the quantification of their accuracy,
precision, recall, and F1 score. Accuracy is calculated by
dividing the total number of correctly predicted instances by the
total number of instances, expressed as a percentage [5, 34],
while precision measures the proportion of true positive
predictions out of all the positive predictions. Recall, on the
other hand, measures the proportion of true positives predicted
by the model out of all the actual positive instances in the
dataset. The F1 score is a harmonic mean of precision and recall,
which helps to balance the two metrics. By using these
statistical measurements, researchers can compare the
performance of different machine learning algorithms and
select the one that provides the best results.
1) Accuracy: It is the ratio of correctly predicted
observations to the total number of observations. The formula
for accuracy is:
Accuracy = (TP + TN) / (TP + TN + FP + FN) × 100
where TP = True Positive, TN = True Negative, FP = False
Positive, FN = False Negative.
2) Precision: It is the ratio of correctly predicted positive
observations to the total predicted positive observations. The
formula for precision is:
Precision = TP / (TP + FP) × 100
3) Recall: It is the ratio of correctly predicted positive
observations to the total actual positive observations. The
formula for recall is:
Recall = TP / (TP + FN) × 100
4) F1 Score: It is the harmonic mean of precision and
recall. The formula for F1 score is:
F1 Score = 2 * ((Precision * Recall) / (Precision + Recall)) × 100
These statistical measurements are commonly used in
machine learning to evaluate the performance of classification
models.
A confusion matrix is a table that is used to evaluate the
performance of a classification model by comparing the
predicted and actual class labels of a set of data [35, 36]. The
matrix contains information about the true positives (correctly
classified instances), false positives (instances that were
classified as positive but were actually negative), true negatives
(correctly classified negative instances), and false negatives
(instances that were classified as negative but were actually
positive) [37]. Fig. 3 displays the confusion matrix for the 2
classes.
Fig. 3 Confusion Matrix
The importance of a confusion matrix lies in its ability to
provide a more detailed evaluation of a classification model's
performance beyond just accuracy. It can be used to calculate
various other evaluation metrics such as precision, recall, and
F1 score, which can be more informative in cases where class
imbalance exists, or the cost of false positives or false negatives
is different. The matrix can also be useful in identifying
patterns in misclassifications, which can be used to further
improve the classification model [36, 38].
IV. RESULTS AND DISCUSSION
We explore various machine learning techniques and their
applicability to solid waste classification tasks, including
supervised learning algorithms such as Decision Trees, Support
Vector Machines (SVM), and K-Nearest Neighbours (KNN).
Additionally, the potential of deep learning approaches, such as
Convolutional Neural Networks (CNNs), for image-based
waste classification is investigated.
The integration of machine learning techniques into solid
waste classification processes can significantly enhance the
efficiency, accuracy, and scalability of waste management
systems. By automating the classification process, machine
learning-based systems can reduce labor costs, minimize
human error, and improve overall waste management outcomes.
Furthermore, the adoption of these advanced technologies can
provide valuable insights into waste generation patterns and
trends, enabling the development of targeted waste reduction
and recycling strategies.
In conclusion, the application of machine learning
techniques for solid waste classification presents a promising
avenue for improving the efficiency and sustainability of waste
management practices. Through the exploration and
development of innovative machine learning-based approaches,
researchers, policymakers, and practitioners can work together
to address the growing challenges of solid waste management
and contribute to a cleaner and more sustainable future.
Table II displays the statistical measures derived from the
machine learning classifiers, considering all 2048 features
which are extracted from image using InceptionV3, along with
the performance time associated with each feature selection.
Without using any feature selection method, the accuracy
obtained by the cubic SVM algorithm was 96.3% in 2592 sec
which is 43.2 mins, while this result slightly decreased and
reached the accuracy of 92.0% in 411.27 sec/ 6.86 mins. The
decision Tree algorithm achieved an accuracy of 85.8% in
627.51 seconds (10.46 minutes). The weighted KNN algorithm
achieved an accuracy of 94.9% in 1404.1 seconds (23.
minutes).
Table III depicts the confusion matrix for machine learning
algorithms. The bar chart in Fig. 4 presents the statistical
measures obtained from three machine learning classifiers,
namely SVM, Decision Tree, and KNN. Without any feature
selection, SVM achieved an accuracy of 96.3%, Decision Tree
achieved 85.8%, and KNN achieved 94.9%. The confusion
matrix involving 2048 features is presented in Table III.
Depending on Table III, the inceptionV3 model was utilized
with 3 distinct machine learning models, to classify organic and
recyclable wastes. According to the confusion matrix
accurately classified 13388 images as organic, and predicted
them as organic, while also correctly identifying 10393 images
as recyclable using SVM with cubic parameter. While the SVM
model misclassified 492 images from organic images and 432
from recyclable once, shown in Table III.
TABLE II
PERFORMANCE METRICS
Performance
Metrics
SVM Decision
Tree
KNN
Feature
Number
2048 2048 2048
Time (mins.) 43.20 10.46 23.
Accuracy (%) 96.3 85.8 94.
Precision (%) 96.8 86.7 94.
Recall (%) 96.8 87.8 96.
F1-Score (%) 96.6 87.2 95.
Fig. 4 Evaluation Metrics Chart for all MLs
TABLE III
MACHINE LEARNING ALGORITHM CONFUSION MATRIX
V. CONCLUSIONS
This study investigates the integration of computer vision
and machine learning technologies to enhance waste
classification and management practices, including identifying
potential challenges and limitations. The research aims to
contribute to developing sustainable and environmentally
friendly waste management solutions by developing an
automated system for accurately and efficiently classifying
waste materials. The study emphasizes the importance of
effective waste management and classification in promoting
sustainability and protecting the planet for future generations.
The future objective is to extend this approach to other types of
solid waste and explore the potential of smart city waste
management.
DATA AVAILABILITY
Contacting the corresponding authors Nnamoko et al.,
(Nnamoko, 2022), or accessing the study's dataset can be found
here; https://data.mendeley.com/datasets/n3gtgm9jxj/2.
REFERENCES
[1] W. Xia, Y. Jiang, X. Chen, and R. Zhao, "Application of machine
learning algorithms in municipal solid waste management: A mini
review," Waste Management & Research, vol. 40, no. 6, pp. 609-
624, 2022.
[2] H.-n. Guo, S.-b. Wu, Y.-j. Tian, J. Zhang, and H.-t. Liu,
"Application of machine learning methods for the prediction of
organic solid waste treatment and recycling processes: A review,"
Bioresource technology, vol. 319, p. 124114, 2021.
[3] M. Koklu, H. Kahramanli, and N. Allahverdi, "A new accurate and
efficient approach to extract classification rules," Journal of the
0.
10.
20.
30.
40.
50.
60.
70.
80.
90.
100.
SVM Decision Tree KNN
Time (mins) Accuracy (%)
Faculty of Engineering and Architecture of Gazi University, vol. 29,
no. 3, pp. 477-486, 2014.
[4] J. Shah and S. Kamat, "A Method for Waste Segregation using
Convolutional Neural Networks," in 2022 Second International
Conference on Advances in Electrical, Computing, Communication
and Sustainable Technologies (ICAECT), 2022: IEEE, pp. 1-5.
[5] Q. Zhang et al., "Recyclable waste image recognition based on deep
learning," Resources, Conservation and Recycling, vol. 171, p.
105636, 2021.
[6] N. Li and Y. Chen, "Municipal solid waste classification and real-
time detection using deep learning methods," Urban Climate, vol.
49, p. 101462, 2023.
[7] O. A. Aworanti et al., "Decoding Anaerobic Digestion: A Holistic
Analysis of Biomass Waste Technology, Process Kinetics, and
Operational Variables," Energies, vol. 16, no. 8, p. 3378, 2023.
[8] A. Shah, V. Patel, and G. Usha, "Employing Machine Learning to
Identify Waste Characteristics," in 2023 International Conference
on Intelligent Data Communication Technologies and Internet of
Things (IDCIoT), 2023: IEEE, pp. 351-356.
[9] H. H. Htun, M. Biehl, and N. Petkov, "Survey of feature selection
and extraction techniques for stock market prediction," Financial
Innovation, vol. 9, no. 1, pp. 1-26, 2023.
[10] E. S. Sabry, S. S. Elagooz, F. E. A. El-Samie, N. A. El-Bahnasawy,
G. M. El-Banby, and R. A. Ramadan, "Evaluation of feature
extraction methods for different types of images," Journal of Optics,
pp. 1-26, 2023.
[11] Y. S. Taspinar, M. Dogan, I. Cinar, R. Kursun, I. A. Ozkan, and M.
Koklu, "Computer vision classification of dry beans (Phaseolus
vulgaris L.) based on deep transfer learning techniques," European
Food Research and Technology, vol. 248, no. 11, pp. 2707-2725,
2022.
[12] G. Uganya, D. Rajalakshmi, Y. Teekaraman, R. Kuppusamy, and A.
Radhakrishnan, "A novel strategy for waste prediction using
machine learning algorithm with IoT based intelligent waste
management system," Wireless Communications and Mobile
Computing, vol. 2022, 2022.
[13] A. S. Girsang, A. D. Saputra, and V. Yanrie, "Performance
Comparison between VGG16 and Inception V3 for Organic Waste
and Recyclable Waste Classification," International Journal of
Intelligent Systems and Applications in Engineering, vol. 11, no. 2,
pp. 557-563, 2023.
[14] A. Altikat, A. Gulbe, and S. Altikat, "Intelligent solid waste
classification using deep convolutional neural networks,"
International Journal of Environmental Science and Technology, pp.
1 - 8, 2021.
[15] N. Nnamoko, J. Barrowclough, and J. Procter, "Solid waste image
classification using deep convolutional neural network,"
Infrastructures, vol. 7, no. 4, p. 47, 2022.
[16] O. Adedeji and Z. Wang, "Intelligent waste classification system
using deep learning convolutional neural network," Procedia
Manufacturing, vol. 35, pp. 607-612, 2019.
[17] G.-W. Cha, W.-H. Hong, and Y.-C. Kim, "Performance
Improvement of Machine Learning Model Using Autoencoder to
Predict Demolition Waste Generation Rate," Sustainability, vol. 15,
no. 4, p. 3691, 2023.
[18] N. B. Nnamoko, Joseph ; Procter, Jack. Waste Classification
Dataset, doi: 10.17632/n3gtgm9jxj.2.
[19] B. Kishore et al., "Computer-aided multiclass classification of corn
from corn images integrating deep feature extraction,"
Computational Intelligence and Neuroscience, vol. 2022, 2022.
[20] S. Hossain, A. Chakrabarty, T. R. Gadekallu, M. Alazab, and M. J.
Piran, "Vision Transformers, Ensemble Model, and Transfer
Learning Leveraging Explainable AI for Brain Tumor Detection
and Classification," IEEE Journal of Biomedical and Health
Informatics, 2023.
[21] N. Alruwais et al., "Hybrid mutation moth flame optimization with
deep learning-based smart fabric defect detection," Computers and
Electrical Engineering, vol. 108, p. 108706, 2023.
[22] R. Butuner, I. Cinar, Y. S. Taspinar, R. Kursun, M. H. Calp, and M.
Koklu, "Classification of deep image features of lentil varieties with
machine learning techniques," European Food Research and
Technology, pp. 1-14, 2023.
[23] Y. Unal, Y. S. Taspinar, I. Cinar, R. Kursun, and M. Koklu,
"Application of pre-trained deep convolutional neural networks for
coffee beans species detection," Food Analytical Methods, vol. 15,
no. 12, pp. 3232-3243, 2022.
[24] A. B. Yilmaz, Y. S. Taspinar, and M. Koklu, "Classification of
Malicious Android Applications Using Naive Bayes and Support
Vector Machine Algorithms," International Journal of Intelligent
Systems and Applications in Engineering, vol. 10, no. 2, pp. 269-
274 , 2022.
[25] I. Cinar and M. Koklu, "Identification of rice varieties using
machine learning algorithms," Journal of Agricultural Sciences, pp.
9 - 9, 2022.
[26] K. Tutuncu, I. Cinar, R. Kursun, and M. Koklu, "Edible and
poisonous mushrooms classification by machine learning
algorithms," in 2022 11th Mediterranean Conference on Embedded
Computing (MECO), 2022: IEEE, pp. 1-4.
[27] R. Grewal, S. Singh Kasana, and G. Kasana, "Machine Learning
and Deep Learning Techniques for Spectral Spatial Classification
of Hyperspectral Images: A Comprehensive Survey," Electronics,
vol. 12, no. 3, p. 488, 2023.
[28] M. Erkinay Ozdemir, Z. Ali, B. Subeshan, and E. Asmatulu,
"Applying machine learning approach in recycling," Journal of
Material Cycles and Waste Management, vol. 23, pp. 855-871, 2021.
[29] I. A. Ozkan, M. Koklu, and I. U. Sert, "Diagnosis of urinary tract
infection based on artificial intelligence methods," Computer
methods and programs in biomedicine, vol. 166, pp. 51-59, 2018.
[30] M. Koklu and K. Sabanci, "Estimation of credit card customers
payment status by using kNN and MLP," International Journal of
Intelligent Systems and Applications in Engineering, vol. 4, no.
Special Issue-1, pp. 249-251, 2016.
[31] T. George, S. P. Potty, and S. Jose, "Smile detection from still
images using KNN algorithm," in 2014 International Conference
on Control, Instrumentation, Communication and Computational
Technologies (ICCICCT), 2014: IEEE, pp. 461-465.
[32] V. Granata et al., "Contrast MR-based radiomics and machine
learning analysis to assess clinical outcomes following liver
resection in colorectal liver metastases: a preliminary study,"
Cancers, vol. 14, no. 5, p. 1110, 2022.
[33] J. Muñoz-Rodenas, F. García-Sevilla, J. Coello-Sobrino, A.
Martínez-Martínez, and V. Miguel-Eguía, "Effectiveness of
Machine-Learning and Deep-Learning Strategies for the
Classification of Heat Treatments Applied to Low-Carbon Steels
Based on Microstructural Analysis," Applied Sciences, vol. 13, no.
6, p. 3479, 2023.
[34] H. Zhang, H. Cao, Y. Zhou, C. Gu, and D. Li, "Hybrid deep learning
model for accurate classification of solid waste in the society,"
Urban Climate, vol. 49, p. 101485, 2023.
[35] M. Koklu, S. Sarigil, and O. Ozbek, "The use of machine learning
methods in classification of pumpkin seeds (Cucurbita pepo L.),"
Genetic Resources and Crop Evolution, vol. 68, no. 7, pp. 2713-
2726, 2021.
[36] R. Kursun, I. Cinar, Y. S. Taspinar, and M. Koklu, "Flower
Recognition System with Optimized Features for Deep Features,"
in 2022 11th Mediterranean Conference on Embedded Computing
(MECO), 2022: IEEE, pp. 1-4.
[37] Y. S. Taspinar, M. M. Saritas, İ. Cinar, and M. Koklu, "Gender
Determination Using Voice Data," International Journal of Applied
Mathematics Electronics and Computers, vol. 8, no. 4, pp. 232-235,
2020.
[38] M. Dogan, Y. S. Taspinar, I. Cinar, R. Kursun, I. A. Ozkan, and M.
Koklu, "Dry bean cultivars classification using deep cnn features
and salp swarm algorithm based extreme learning machine,"
Computers and Electronics in Agriculture, vol. 204, p. 107575,
20 23.