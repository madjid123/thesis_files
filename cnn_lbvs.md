
# Deep Convolutional Neural Network to improve the performances of screening process in LBVS
Keywords:
Deep learning
Convolutional neural network
Molecular descriptors
Similarity coefficients
Ligand-based virtual screening
Similarity searching
Drug discovery
2D fingerprint

## ABSTRACT
Drug design is a research process with a goal of creating a chemical drug to produce the desired biological
effect. Because of the long time and the high cost issues associated with traditional drug discovery, there
is a need to develop new techniques and strategies to increase the diminishing effectiveness of traditional
approaches. Ligand-Based Virtual Screening (LBVS) plays a vital role in the early stage of the drug discovery.
It could constitute a possible solution to solve the time and cost problems. Subsequently, the researchers are
looking for new methods to find new active compounds and bring them to market in a short time. LBVS can
be enhanced by different methods and strategies such as Machine Learning and Deep Learning.
In this paper, a Deep Convolutional Neural Network method is proposed to enhance the performances of
Ligand-Based Virtual Screening process (DCNNLB). Two main contributions are presented in this paper, The
first contribution consists of designing a model based on Deep Convolutional Neural Network (DCNN) for
LBVS. We propose several topological network models to find the one that gives the best performance such
as accuracy and recall. For this, many network topology configurations have been proposed, and a variety of
parameters have been taken into account. Furthermore, our proposed model is trained on all compounds of
all activity classes of the Drug Data Report Database (MDDR). Thus, it presented a mean accuracy of 0.98 for
all three MDDR Datasets. The second contribution is to generate a new learning representation in order to
better represent chemical compound. This representation is based on the extraction of the automatic features
learning from the weights of our proposed model. Consequently, it is very efficient in calculating molecular
similarity and performances of the LBVS process. The obtained results with the three different datasets drawn
from the MDDR and the performance evaluation with ANOVA test, have proved the superiority in performance
of our proposed method compared to the different conventional methods.

1. Introduction
Virtual screening is the process of selecting chemical compounds or
molecules to facilitate bioactivity testing. Ligand-based virtual screening extrapolates from known active compounds that are used as input information, and aims to identify structurally diverse compounds
with similar bioactivity, regardless of the methods applied. The virtual
screening methods can be used in many aspects of bioinformatics
domain, such as docking, molecular classification, clustering, and the
prediction of biological activity of compounds. Similarity searching is
the simplest method used in Virtual Screening. It is mainly based on
Similar Property Principle (Rawlins, 2004; Willett et al., 1998). This
last states that; the molecules having the same chemical structure tend
to have similar biological and physicochemical (Johnson & Maggiora,

1990). The increased importance of similarity search applications and
the improvement of the performance of the screening process in the
LBVS is in part due to their important roles in the lead optimization
phase of drug discovery. Therefore, this improvement in performance
allows to increase the number of active molecules that will contribute
in the design of new drug candidates.
The effectiveness of similarity searching in LBVS can be improved
by using different techniques. In the recent years, Deep Learning (DL) is
considered as a sub-domain of machine learning (ML) which has grown
enormously. This is mainly due to the success gained in the several
fields such as voice recognition, text mining (Su & Lu, 2017), object
detection, image recognition (Atto et al., 2020; Wu et al., 2020), and
many other domains such as genomics and drug discovery (Sun et al.,

∗ Corresponding author at: Faculty of Technology, University of SETIF 1, Setif, Algeria.

E-mail addresses: berrhail.fouaz@univ-setif.dz (F. Berrhail), hacene.belhadef@univ-constantine2.dz (H. Belhadef), mohammed.haddad@univ-lyon1.fr
(M. Haddad).
https://doi.org/10.1016/j.eswa.2022.117287
Received 3 June 2021; Received in revised form 29 March 2022; Accepted 21 April 2022
Available online 30 April 2022
0957-4174/© 2022 Elsevier Ltd. All rights reserved.

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

data fusion and the combination of many different measures of similarity and descriptors have been suggested for chemical structure matching, each focusing on a particular type of molecular feature. Its aim is to
improve the performance of the search process in chemical databases,
researchers have begun to investigate the benefits of combining the
results of different measures of similarity. Berrhail et al. (2017) and
Fouaz et al. (2019) investigated the effect of using different combinations of fingerprints and similarity coefficients for similarity searching
in LBVS. The obtained results in this work, demonstrate that the performance of some combinations with some coefficients is superior to the
performances obtained in combinations with the Tanimoto coefficient.
Machine learning has been used in virtual screening and QSAR process,
Kauffman and Jurs (2001) used the k-NN approach with QSAR as a
potential screening mechanism for larger libraries of target compounds,
while the prediction for the k-NN method was based on the property
of the molecules reported by Konovalov et al. (2007). Abdo and his
collaborators (Abdo & Salim, 2009) have proposed new methods for
similarity searching using Bayesian Inference Networks (BIN), and his
experimental results outperform all conventional methods. In addition,
and in order to improve the Bayesian network, they also used fragment
reweighting techniques for the selection of characteristics (Ahmed
et al., 2012a., 2012b).
Deep Learning (DL) are a recent extension of Artificial Neural
Networks (ANNs), which use deep and specialized architectures to
learn useful functionality from raw data (Lo et al., 2018). Recently,
the success of deep learning provides the opportunity to develop new
techniques and tools that can automatically retrieve representations
of chemical structures (Lo et al., 2018; Sun et al., 2017). According to Wang and Raj (2017), DL approaches are more efficient than
conventional ML tools because they also include the method for extracting characteristics. The Convolutional Neural Network is a type
of feed-forward deep learning network that can be easily trained and
generalized to other networks with connectivity between adjacent layers (Krizhevsky et al., 2017). The CNNs are also very compatible with
the QSAR representations of compounds, which means that they are
applicable to virtual screening. One of the first CNNs used for virtual
screening was AtomNetTM, developed by Atomwise, Inc. (Wallach
et al., 2015). In contrast to the majority of ML-based virtual screening,
Atom-NetTM uses an SBVS, which works well with the convolution
capability to extract local feature sets from multidimensional data.
Recently, various works based on the techniques of textual information retrieval and feature selection are presented. The quantum-based
similarity (SQB) measure was used in the work of Al-Dabbagh et al.
(2015), offering a clear enhancement for the LBVS. Thereafter, in
an other work (Al-Dabbagh et al., 2017) he developed a new approach to rank molecular compounds called the Quantum Probability
Ranking Principle (QPRP), which is based on the quantum mechanics. The QPRP ranking criteria would allow to establish an analogy
between the molecular structure ranking process and the physical
experiment for 2D fingerprints in LBVS. The experimental results surpassed the conventional methods. In addition, several works which
uses feature selection techniques, Berrhail and Belhadef (2020) have
proposed an approach for selecting characteristics on the basis of the
genetic algorithm (FSGASS: Feature Selection using Genetic Algorithm
for Similarity Searching in LBVS). The FSGASS allowing at first; the
reduction in features space, the elimination of redundancy and the
decrease in training run-time, and secondly boost the performances
and the effectiveness of the screening process. The obtained results
have demonstrated a superiority in performance compared to the conventional methods. Swarm intelligence (IS) has been proven to be a
strategy that solves problems related to different optimizations including NP-hard problems. It has been used for the selection of features in
different applications (Brezočnik et al., 2018).
Finally, after having studied and analysis all these different approaches and methods, we can notice that: each approach has its
advantages and drawbacks, and each of which uses its own strategies

2017). The recent success of deep learning (DL) offers the opportunity
to develop algorithms and tools to extract automatically new representations of specific structures. Deep learning technology has undergone
massive developments, empirical results have shown that this technique
is better than the other classical machine learning algorithms. In addition, new techniques that integrate chemical molecules into Deep
Learning architectures for LBVS and for the predicting of the biological
activities of chemical compounds need to be developed. Consequently,
our main research questions in this work are: Can the use of Deep
learning techniques, in particular Deep Convolutional Neural Network,
affect and improve the performance of the LBVS and the process of
predicting the biological activity of molecules?, and how to find and
generate on the basis of a Deep Convolutional Neural Network model
an efficient molecular representation which allows to better represent
the molecules?
In this work, we are mainly interested in improving the virtual
screening process based on ligands. For this reason, our main objective
is to propose a new method which is mainly based on Deep Convolutional Neural Network to improve the performance of LBVS process.
This method namely: Deep Convolutional Neural Network for LigandBased Virtual screening (DCNNLB). We have presented in this paper
two main contributions; the first contribution consists of proposing
and designing a model based on deep Convolutional Neural Network
(DCNN) for the prediction of biological activity of molecules. This
model is mainly composed of several layers: convolution and Maxpooling, Dense and softmax layer. An experimental study is carried out
for the configuration of the architecture of our proposed model and
the optimization of its parameters such as: the activation function, the
size of convolution filter and the number of layers used. The second
contribution is to generate a new learning representations in order to
better represent chemical compounds. This representation is based on
the extraction of the automatic features learning from the weights of
our proposed Model.
Our approach consists of using 2D molecular fingerprint representation, which is generated from software such as Pipeline Pilot. The
molecular 2D fingerprints are a means of encoding the structure of a
molecule. In addition, The Com2Mat (Compound to Matrix) technique
is used to represent and transform the chemical compound into a matrix
format based on their 2D fingerprint. These obtained matrix representations are the most suitable input data for our proposed DCNN Model.
Moreover, In order to design our model, we propose several topological
network models to find the one that gives the best performance such
as accuracy and recall. For this, many network topology configurations
have been proposed, and a variety of parameters have been taken into
account, Furthermore, our proposed model is trained on all compounds
of all activity classes of the MDDR datasets. New automatic learning
representations of compounds have been generated. These molecular
representations are very efficient and very useful for calculating the
molecular similarity and the performance of the ligand-based virtual
screening process. The ANOVA test was used to evaluate and compare
the results obtained by our proposed method DCNNLB with the results
obtained by other conventional methods such as Tanimoto method.
2. Related works
In the literature, different methods and approaches to improve the
performance of LBVS and the biological prediction process can be cited.
Several research studies use weighting schemes for better recall and
accuracy. They assume that molecular fragments that are not related
to biological activity have the same weight as important fragments.
It is common for chemists to consider some characteristics as more
important than others through chemical structure diagrams, such as
functional groups. It is therefore more important to give more weight
to these descriptors than to others; examples of these works (Abdo &
Salim, 2011; Karnachi & Brown, 2004; Whittle et al., 2003). The use of


F. Berrhail et al.

and techniques to enhance the performance of prediction and the
screening process for molecular similarity searching. We can distinguish a few classes of approaches based on different classification
criteria such as: approaches based on scheme weighting, approaches
coefficients-based similarity, approaches of data fusion, approaches of
nonlinear similarity (machine learning techniques, deep learning and
Bayesian-based similarity searching), approaches based on mathematical model to describe new similarity method, approaches based on text
retrieving techniques, and finally, the features selection approaches,

a: Bits set to ‘1’ in both Ma, Mb.
𝑛
∑

𝑎=

𝑊𝑗𝑎 .𝑊𝑗𝑏

(2)

𝑗=1

b: Bits set to ‘1’ exclusively in Ma..
𝑛
∑

𝑏=

𝑊𝑗𝑎

(3)

𝑗=1

c: Bits set to ‘1’ exclusively in Mb

3. Methods

𝑛
∑

𝑐=

𝑊𝑗𝑏

(4)

𝑗=1

3.1. Virtual screening process

d: Bits set to ‘0’ in neither Ma or Mb

Virtual Screening (VS) refers to the use of computer methods to
process compounds from a library or database of chemical compounds
in order to select and identify those that are likely to possess a desired biological activity, such as the ability to inhibit the action of a
particular therapeutic target (Bajusz et al., 2015; Jorissen & Gilson,
2005; Willett, 2013). Virtual screening is based on many concepts
and techniques, such as the important basic idea which is the Similar
Property Principle. This principle is considered to be the core of the
screening orientation and represents one of the basic steps used in
many drug discovery applications such as compound selection, virtual
screening and targeted design of libraries.
Recently, several methods and approaches for measuring the structural similarity of the chemical compounds and enhancing their performance in LBVS have been introduced. The complexity of measuring
similarity is one of the most important issues when quantifying similarity between two compounds. This complexity is highly dependent
on the representation used to describe the molecule and the similarity
coefficient used to measure similarity. In this work, we have made the
choice on the 2D fingerprint representations to describe the chemical
compounds. This choice is motivated by the ability of the 2D fingerprints to differentiate between different compounds in terms of their
physicochemical, bioactivity, or any other properties. In addition, they
optimize the structuring of the chemical space, and they allow a good
representation of the chemical reality of the chemical system.
Improving the performance of similarity searching in LBVS, which is
the subject of this work, involves the use of similarity coefficients. Thus,
different similarity coefficient approaches have also been presented and
many types of similarity measurements have been introduced. However
similarity measurements based on the 2D fingerprint with a simple
association coefficient Tanimoto are by far the most common, and the
most popular.

𝑑=

𝑎
𝑎+𝑏+𝑐

1 − 𝑊𝑗𝑎 − 𝑊𝑗𝑏 + 𝑊𝑗𝑎 .𝑊𝑗𝑏

(5)

𝑗=1

n: Number of features (bits) in the both bit-strings of Ma and Mb
𝑛=𝑑+𝑎+𝑐+𝑏

(6)

Eq. (7) Tanimoto formula for continuous data:
We consider A and B two molecules represented by vectors,
𝑋𝐴 𝑎𝑛𝑑 𝑋𝐵 respectively, of length n features.
∑𝑛
𝑖=1 𝑋𝑖𝐴 .𝑋𝑖𝐵
𝑆𝐴,𝐵 = ∑ (
(7)
)2 ∑𝑛 (
)2 ∑
𝑛
+ 𝑖=1 𝑋𝑖𝐵 − 𝑛𝑖=1 𝑋𝑖𝐴 .𝑋𝑖𝐵
𝑖=1 𝑋𝑖𝐴
The Ligand-Based Virtual Screening process plays a vital role in the
early stage of the drug discovery, it could constitute a possible solution
to solve the time and cost problems. Subsequently, the researchers are
looking for new strategies to find new active compounds and bring
them to market in a short time. The main elements of the virtual screening process and the prediction of the biological activity of compounds
include the representations (descriptors), which are used to translate
chemical structures into mathematical variables. Computational approaches, represent an economic alternative for virtual screening and
property prediction. However, the irregular error rate and the limited
performance of these predictions limit their use in the research process.
3.2.

Deep convolutional neural network

Machine learning methods, in particular artificial neural networks
(ANNs), have been used and applied for virtual screening and prediction of biological activity of compounds for a long time. ANNs
are considered one of the best known machine learning techniques
because of their perceived ability to mimic the activities of the human
brain, albeit in a simple way (Zurada, 1992). The QSAR (Quantitative
structure–activity relationship) method is used to establish a quantitative relationship between the molecular structures of the compound
and its biological activity. There are no direct methods to determine the
activities of chemical compounds based on their structure. Therefore,
indirect methods must be used to generate mathematical models that
describe this relationship. In these techniques, certain mathematical
algorithms are used to calculate the molecular representations that are
used rather than the primary chemical structure.
The Deep Learning technology has undergone massive developments during the past few years. Empirical results showed that this
technique was better than the other ML algorithms. This could be
due to the fact that this technique mimics the brain functioning and
stacks multiple neural network layers one after another, like the brain
model. Therefore, Neural networks and deep learning have also been
successfully applied in the field of chemoinformatics through creative
manipulations of the 2D fingerprint representation of chemical structure and the construction of the network architecture are considered as
alternatives to intensive quantum chemical calculations (Ragoza et al.,
2017).

3.1.1. Tanimoto model for similarity measures in LBVS
Similarity coefficients are used to obtain a numerical quantification of the degree of similarity between a pair of molecules. These
coefficients are simply a set of functions or formulas that are used
to transform the differences between a pair of compounds into a real
number, usually in the range of units [0–1] (Ahmed et al., 2012b).
According to the research work (Bajusz et al., 2015; Whittle et al.,
2006; Willett, 2013) the Tanimoto similarity coefficient is currently
considered as the conventional method and the most used coefficient
in chemical information systems of similarity based on 2D fingerprints.
In the literature, two formulas were employed for binary and continuous data, as presented in Eqs. (1) and (7), respectively:
Eq. (1) Tanimoto formula for binary data:
We consider Ma, Mb: two compounds described by means of vectors
Wa and Wb of length n features.
𝑆𝑀𝑎,𝑀𝑏 =

𝑛
∑

(1)

where:
3

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 1. An example of typical Deep CNN model for LBVS.

The applications of deep convolutional neural network in LBVS are
still in their initial phases, thus an enormous work and investigations
are recommended to propose new methods and techniques based on
deep learning for solving problems of LBVS process. As illustrated in
Fig. 1, a typical Deep CNN consists of many different layers, starting
usually in the first step with the convolutional layer. This layer receives
as input the data of chemical compounds in the form of 2D fingerprints
after its transformation into a new representation (Com2Mat representation). This layer is then responsible for the convolution of the
molecular representation, and the generation of feature maps which
preserve the spatial locality of the features. Therefore, the feature maps
done by dragging a group of small filters (‘‘kernels’’), each containing a
number of learnable weights on the input molecular representation and
summing the term-to-term multiplications at each possible position.
The feature map generated by each filter, is a new layer and contains
the results of the particular filter in the input molecular representation.
In the next step, the resulting group of layers is subjected to a subsampling (pooling) process in which sets of elements in the feature maps
are combined and reduced to a single value based on a given criterion
(for example, average pooling, max pooling..etc.). A Deep CNN architecture, can contain an alternative of convolution and pooling layers
using different parameters such as filter size, depth..etc. This allows
the successive extraction of higher level features. This extraction is
considered as one of the strong points of CNNs. As a finale step, the
last subsampling layer can be reduced to a single vector containing all
its weights and linked to a fully connected layer. This letter is linked
to the output layer which contains a field for each possible class and
provides us with the estimated classification and the prediction of the
biological activity of the input compound in our DCNN model.

Our general computational framework using Deep Convolutional
Neural Networks for chemoinformatics data analysis, compound activity prediction and virtual screening, is illustrated in Fig. 2. The first
step of the chemoinformatics analysis consists in extracting the characteristics of the compounds, thus, this phase allows to characterize the
compounds by representations or chemical descriptors using different
software such as Pipeline Pilot. The chemical characteristics of the
compound are represented by chemical 2D fingerprints and applied
to compare the similarity of the compound based on the presence
and the absence of common chemical characteristics. The chemical
fingerprint is transformed into another representation (Com2Mat) to
feed and train our deep learning CNN model for predicting other
chemical, physiochemical and biological properties. Thereafter, a new
representation is generated based on the parameters of our learning
model to characterize all the compounds in the dataset. In addition,
this new molecular representation is very useful in similarity search
and ligand-based virtual screening. Finally, a performance evaluation
and comparison of results is performed, which allows for more accurate
and reliable results.
3.3.1. Input compounds representations for our DCNN model
Having good input features is considered one of the major issues
affecting the performance of similarity searching and virtual screening
process. 2D fingerprints are a specific type of complex descriptor that
can identify the disposition of features from the bit string representations (Whittle et al., 2006). The molecular fingerprints are an important
tool that can be used for the biological activity prediction and virtual
screening process (Ahmed et al., 2012b; Al-Dabbagh et al., 2017;
Berrhail & Belhadef, 2020; Berrhail et al., 2017; Cereto-Massagué et al.,
2015; Fouaz et al., 2019). They were seen to be popular due to their
simplicity, easiness of application and calculation speed, which are
important while screening the virtual libraries consisting of millions
of molecules. The fingerprints refer to a gross simplification of all
molecules, which help in comparing and detecting ‘‘similar’’ molecules.
The features extraction stage is very important for data analysis in
deep learning process. This step identifies the interpretable representation of data for the machines that can improve the performance of
these learning algorithms. The application of inappropriate features
can decrease the performance of the best learning algorithms, while
simple techniques can perform very well if the appropriate features are
applied. Moreover, the screening process could be slowed due to a high
number of features while giving similar results as obtained with a much
smaller feature subset (Berrhail & Belhadef, 2020). In this work, we
have proposed a technique that could predict the biological activities
of the molecules with the help of the molecular fingerprints in our Deep
Convolution Neural Network model. This technique offers proposals
and implementation details regarding the pre-processing step that could
be used to detect biological activities. In addition, we have investigated
the 2D fingerprints ECFP4 which included 1024 features of compound.
It was generated by using the Scitegics Pipeline Pilot software (Biovia,
2021). We present below a summarized Algorithm that we have used
to transform the storage of the fingerprint representation for every

3.3. The proposed Deep Convolutional Neural Network for Ligand-Based
Virtual screening (DCNNLB)
In this work, we present our new proposed method based on Deep
Convolutional Neural Network model to improve the performance of
LBVS. The implementation of Deep Learning in the LBVS requires further a high investigation in the representation of chemical compounds.
Molecules are re-represented to match the fundamental principles of
the convolutional neural network, such as the use of the matrix format
to represent molecular structures in the CNN architecture. Convolutional neural networks (CNNs) are a type of deep feed-forward network
commonly used in image recognition. They can be easily trained and
generalized as compared to other networks having connectivity between the adjacent layers (Dahl et al., 2013). CNNs hierarchically
decompose the representation of compound so that each layer of the
network learns to recognize higher-level features while maintaining
their spatial relationships. The Com2Mat (Compound to Matrix) is
technique that has been used to represent and transform the chemical
compound into a matrix format based on their 2D fingerprint representation. These obtained matrix representations are the input data of our
DCNN model.
4

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 2. Computational framework using deep convolutional neural networks for chemoinformatics data analysis, compound activity prediction and virtual screening.

molecule to the Matrix representation (Com2Mat) using the row-major
order. Thus, this is done by applying the formula (8) for all compound
fingerprints in the data sets. A representation is shown in Fig. 3 of the
Com2Mat representation.
𝐶𝑜𝑚2𝑀𝑎𝑡[𝐼, 𝐽 ] = 𝑉 𝑒𝑐𝑡𝑜𝑟_𝑓 𝑖𝑛𝑔𝑒𝑟𝑝𝑟𝑖𝑛𝑡[𝑆 ∗ (𝐼 − 1) + 𝐽 ]

8.
9.
10.

(8)

ForEnd
ForEnd;
ForEnd;

3.3.2. Convolution and subsampling layers
Convolutional Neural Networks has one input and one output layer.
The number of hidden layers differ between different networks depending upon the complexity of the problem to be solved. The convolutional
layers use the convolution operation for the input and pass the output
to the subsequent layer. A convolutional layer contains a set of filters
whose parameters must be learned. The weight and the height of the
filters are smaller than those of the input 2D molecular representations (Com2Mat). Therefore, every filter is convolved with the input
representation to calculate an activation map made of neurons. The
filter is dragged across the width and the height of the input layer
and the products of the points between the input and the filter are
calculated at each spatial position. The output of convolutional layer
is obtained by stacking the activation maps of all filters along the
depth dimension. Since the width and the height of each filter are
designed to be smaller than the input representation, each neuron in
the activation map is connected to only a small local region of the
input volume. After each convolutional layer, a non-linearity layer is
introduced which allows to be introduced the non-linearity into the
system that calculates the linear operations in the convolutional layers.
For this purpose, different non-linear functions are used such as step,
tanh, sigmoid, ReLu. The pooling layer allows multi-scale analysis and

where : S represents the size of matrix representation for every fingerprint of compound, and I, J represents the Com2Mat matrix index
position.
Algorithm 1: Input
D: Dataset of compounds;
n: Number of features of compound in DB;
m: Number of compound in DB;
Vector_fingerprint: 2D fingerprint;
N: size of Com2Mat matrix; I,J,L,R: integer variables;
Output:
Com2Mat: Matrix of representation
𝐶𝑜𝑚2𝑀𝑎𝑡 ⟵ 𝐼𝑛𝑖𝑡𝑖𝑎𝑙𝑖𝑧𝑒_𝑟𝑒𝑝𝑟𝑒𝑠𝑒𝑛𝑡𝑎𝑡𝑖𝑜𝑛(𝑁, 𝑁)
0. 𝐹 𝑜𝑟(𝑅 ⟶ 1..𝑚; 𝑅 + +)
1.
𝑉 𝑒𝑐𝑡𝑜𝑟_𝑓 𝑖𝑛𝑔𝑒𝑟𝑝𝑟𝑖𝑛𝑡 ⟵ 𝐷𝐵(𝑅)
2.
𝐹 𝑜𝑟(𝐼 ⟶ 1..𝑁; 𝐼 + +)
3.
𝐹 𝑜𝑟(𝐽 ⟶ 1..𝑁, 𝐽 + +)
4.
𝐿 ⟵ 𝑁 ∗ (𝐼 − 1) + 𝐽 ;
5.
𝐼𝑓 (𝐿 ≤ 𝑛)
6.
𝐶𝑜𝑚2𝑀𝑎𝑡(𝐼, 𝐽 ) ⟵ 𝑉 𝑒𝑐𝑡𝑜𝑟_𝑓 𝑖𝑛𝑔𝑒𝑟𝑝𝑟𝑖𝑛𝑡(𝐿);
7.
IfEnd;
5

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 3. Summarized design to transform the fingerprint representation to Com2Mat representation.
Table 1
Structure of DS1 dataset.

reduces the input size of the 2D matrix representations of compounds.
Standard pooling operators include maximum pooling and average
pooling, which help to calculate a maximum or average value in the
small spatial block.
3.3.3. Design architecture of our DCNN model
In this section, we describe the development of our DCNN model for
similarity searching in LBVS, which is trained to predict the biological
activity of all compounds in chemical database using Com2Mat representations of all compounds. As previously mentioned, the molecular
descriptors 2D fingerprints were generated using Scitegics Pipeline
Pilot software. These fingerprints were then stored in the 2D Matrix
(𝑁 ∗ 𝑁) with a row major order to derive a new Com2Mat matrix
representation, which characterized the various molecular properties.
A convolutional architecture with fully connected layers was considered as the default architecture of our work. To design the architecture of our DCNN model, we have used the principles of the
configuration from the generic configuration presented in Krizhevsky’s
work (Krizhevsky et al., 2017) and Wang and Raj (2017). The generic
architecture of our DCNN model, in which the molecules were passed
through the stack of one or more convolutional (Conv2D) layers. These
convolutional layers applied several kernel filters; a max-pooling layer
was used in the convolution step. It was seen that this combination (convolution, max pooling) improved the CNN configuration and
helped our model to enhance the effectiveness of LBVS and yielded
the best activity prediction for all chemical compounds. The maxpooling layer was followed by the flattened layer. This converted
the 2D matrix data representation into one vector, which helped in
processing the output having fully connected layers, called the dense
layers. Furthermore, the regularization layer made use of dropouts
and was configured in a manner which randomly excluded 50% of
all neurons, to decrease the overfitting. The final layer was made of
the Softmax layer. In our study, we propose several topological models
of networks in order to find the one that gives the best performances
such as accuracy and recall. Numerous network topology configurations
were proposed, and a different variation in the number of convolution
layers (Conv2d) and max-pooling layers has been established for each
CNN topological models. Subsequently, for each topological model,
we have applied different configuration parameters such as kernel
size, activation function, size of the pooling window. Finally, all CNN
configuration models have the same terminal layers, which are flatten,
dropout, and softmax.

Index

Activity class

Active molecules

31 420
71 523
37 110
31 432
42 731
6233
6245
7701
6235
78 374
78 331

Renin inhibitors
HIV protease
Thrombin inhibitors
Angiotensin II AT1antagonists
Substance P antagonists
Substance P antagonists
5HT reuptake de inhibitors
D2 antagonists
5HT1A agonists
Protein kinase C inhibitors
Cyclooxygenase inhibitors

1130
750
803
943
1246
752
359
395
827
453
636

Table 2
Structure of DS2 dataset.
Index

Activity class

Active molecules

7707
7708
31 420
42 710
64 100
64 200
64 220
64 500
64 350
75 755

Adenosine (AI) agonists
Adenosine (A2) agonists
Rennin inhibitors 1
CCK agonists
Monocycle-lactams
Cephalosporin’s
Carbacephems
Carbapenems
Tribactams
Vitamin D analogues

207
156
1300
111
1346
113
1051
126
388
455

as antihypertensives and others to specific enzymes such as renin
inhibitors. The MDDR database is characterized by a limited set of
activities from which to choose, unlike other databases. In this research
work, subsets of the MDDR database were used in the experiments, and
these datasets were organized to test the performance of our proposed
methods. In our work, we have used the MDDR databases that are
represented by the Extended Connectivity Fingerprints of the circular
substructure ECFC4 (Extended Connectivity Fingerprint Counts). The
fingerprints ECFC4 are circular topological fingerprints designed for
molecular characterization, structure–activity modeling, and similarity
search. They have been used in many virtual screening applications.
The variation of extended connectivity fingerprints provided by the
Pipeline Pilot software (Biovia, 2021). The MDDR database contains
three data sets : DS1, DS2 and DS3. The DS1 contains eleven classes
of activity, including two types of active compounds (heterogeneous
and homogeneous). DS2 can be distinguished from DS1 because it has
ten homogeneous activity classes. In the DS3 dataset, there are ten
heterogeneous activity classes. The structures of the three datasets are
illustrated in Tables 1–3. In each row of a table, the activity index, the
number of active molecules in the activity class and the name of the
activity are given.

4. Experiments
4.1. Datasets
There are a number of authorized databases that attribute biological activities to compounds. However, the MDDR database (Benedict,
2021) remains one of the most popular databases in chemoinformatics,
which has been compiled from patent literature since 1988. The MDDR
database contains over 102,000 chemical compounds with several distinct activities, some of which are related to therapeutic areas such

4.2. Configuration of the proposed deep CNN model for LBVS
In this section, as mentioned previously, we have proposed several
network topologies to find a learning model that can initially provide
6

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.
Table 3
Structure of DS3 dataset.
Index

Activity class

Active molecules

9249
12 455
12 464
31 281
43 210
71 522
75 721
78 331
78 348
78 351

Muscarinic (M1) agonists
NMDA receptor antagonists
Nitric oxide synthase inhibitors
Dopamine-ydroxyl inhibitors
Aldose reductase inhibitors
Reverse transcriptasede
Aromatase inhibitors
Cyclooxygenase inhibitors
Phospholipase A2 inhibitors
Lipoxygenase inhibitors

900
1400
505
106
957
700
636
636
617
2111

Concerning the results obtained to test and evaluate different CNN
model topology illustrated in Table 8; we can see that the model
labeled J has a superiority in Mean accuracy for all data sets (Mean
accuracy=0.9897). Therefore, it can be considered as the proposed
model for similarity searching in ligand-based virtual screening. Thus,
this model is mainly composed of three convolution layers with Relu
activation functions, and a Max-pooling layer for sub-sampling. The
Max-pooling layer is characterized by the maximum value of the square
regions which is of size (2 ∗ 2). The max-pooling layer is followed by
the flattened layer; which converted the 2D matrix data into one vector.
This latest help in processing the output having fully-connected layers,
called the dense layers. Furthermore, the regularization layer made use
of dropouts and is configured in a manner which randomly excluded
50% of all neurons, to decrease the overfitting. The last layer is made of
the Softmax layer. All details of the proposed CNN model for similarity
searching in LBVS are showed and summarized in Fig. 4 and Table 9.

Table 4
The obtained results of testing model using different activation functions for all MDDR
data sets.
Activation function

softplus

sigmoid

tanh

linear

relu

Accuracy for DS1
Accuracy for DS2
Accuracy for DS3
Mean

0.9347
0.9488
0.8230
0,9022

0.1688
0.9407
0.6109
0,5735

0.9323
0.9597
0.9287
0,9402

0.9469
0.9555
0.8982
0,9335

0.9601
0.9698
0.9330
0,9543

## 5. Results and discussion
In this part, we present our results obtained by the application of
our proposed DCNN model for molecular similarity searching in LBVS.
Firstly, to evaluate the performance of our DCNN model, we have
used all the activity classes of the MDDR datasets. The data set DS1
contains 8294 molecules that are distributed over the eleven activity
classes. The data set DS2 contains 5083 molecules distributed over ten
activity classes, and the last data sets contains 8568 molecules that
are distributed over the ten activity classes. In addition, our model is
evaluated and validated on these data sets, taking into account for each
data set a percentage of 0.70 for training and 0.30 for validation. We
have considered the accuracy as an evaluation metric, and we have set
the error loss function (losses.categorical_crossentropy of keras) (keras,
2022), which is a measure of the error taken by our model. This error
is defined in Keras API, which is a deep learning API written in Python,
and executed in most of the deep learning platform TensorFlow (keras,
2022). Consequently, both metrics are calculated at each epoch during
the training and testing stages. The results obtained from the evolution
of the accuracy value and the loss value for each data set for the
training and testing data are shown graphically in Figs. 5–10.
In the graphical results for all data sets, we can see that the accuracy
values increase drastically in the first epochs, and then increase more
and more afterwards until the maximum value is reached. For the
error loss function, we find that in all data sets a drastic decrease
in the first epochs, and then it has been steadily decreasing slowly.
Additionally, we have noticed a slightly overfitting issue in both DS1
and DS2 data sets. This is justified by the type and the representation
of compounds that they contain, DS1 comprises heterogeneous and
homogeneous compounds, while DS3 only comprises heterogeneous
compounds. For this purpose a dropout layer is put in place in order to
solve this problem.
The learning molecular representation is the new resulting representation by extracting the weights of the 380 neurons of our network, in
particular the fully connected layers. The datasets used for the virtual
screening are fed by our network model, which is fully saved. Thus, our
interest is to investigate and determine if this new representation would
allow to improve the performance of the LBVS. We have illustrated in
the Figs. 11–13, the scatter plots using all molecules that have been
classified into different classes of biological activity of the MDDR data
sets (DS1, DS2, DS3) respectively, based on the new learning representation using PCA (Principal Component Analysis). Principal component
analysis is a mathematical algorithm that reduces the dimensionality
of the data while retaining most of the variation in the data set. PCA
identifies new variables, the principal components, which are linear
combinations of the original variables (Ringnér, 2008). The principal
components are normalized eigenvectors of the covariance matrix and
ordered according to how much of the variation present in the data
they contain. Each component can then be interpreted as the direction,

better performance for the process of predicting the biological activity
of compounds. In a second step, it also allows to generate better
learning representations of chemical compounds, which is based on
the weights of our network. Thus, several configurations have been
used, based on a variation of the different parameters of CNN architecture such as the number of convolution layers (Conv2D), number of
Max-pooling layers, activation function, kernel size...etc. In addition,
All experiments in this work were performed in python (3.5) with a
system configuration of Intel(R) Xeon(R) W-2123 CPU @ 3.60 GHz,
32 GB RAM, and a Windows 10 operating system −64 bits. Our first
experiment is to study the effect of choosing the activation function
for our proposed DCNN model. For this purpose, we have chosen five
activation functions in order to test them on a simple generic CNN
model using the three MDDR data sets (DS1, DS2, and DS3). These
functions are: softplus, sigmoid, tanh, linear, and relu. The results of
the evaluation and testing are presented in Table 4.
Related to the obtained results of using different activation functions. The relu function demonstrated a superiority in performance with
mean accuracy of 0,9543 for all MDDR data sets, so we will use the
relu function in all next experiments. The second experiment consists
in studying the effect of choice of the size of convolution layers kernels,
so two kernel sizes have been chosen (size value: 3 and 5). Thereafter,
to test and investigate these kernel sizes, we propose six configurations
of CNN model. All further references to the configurations will be
made based on their labels (A–F). We have varied the number on
the convolution layers for each kernel size (as one, two and three
layers). Table 5 describes the configuration details of each model, and
Table 6 illustrates the obtained results of the evaluation and testing
using different kernel size and different numbers of 2DConv layers for
all MDDR datasets.
As mentioned above in Table 5, we have varied the configuration
of the six CNNs. For each of the three models, the kernel size is fixed
and the number of layers is varied. The obtained results from the
experimentation of the proposed models show a superiority in performance for the model C. After setting the optimal parameters of the CNN
architecture configuration such as kernel size and activation function,
we want to study in the third experiment the effect of choosing a variety
of CNN network topologies. Thus, different configurations were used,
the number of convolutional layers varied from 1 to 6 with one or
more subsampling layers in each CNN. Each CNN model configuration
is labeled by G–M. The configuration details for each model CNN is
showed in Table 7, and the obtained results of testing model using
different CNN model topology are showed in Table 8.
7

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 4. Architecture of the proposed Deep CNN model for LBVS and activity prediction.

Fig. 5. Variation of accuracy value for training and testing data in DS1 data set.

Fig. 6. Variation of loss value for training and testing Data in DS1 data set.

Fig. 7. Variation of accuracy value for training and testing data in DS2 data set.

8

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.
Table 5
CNN configuration details for varying the number of convolutional layers and kernel sizes.
CNN Model configuration
A

B

C

D

E

F

One layer

Two layers

Three layers

One layer

Two layers

Three layers

2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)

2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)
2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)

2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)
2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)
2𝐷𝐶𝑜𝑛𝑣(3 ∗ 3)

2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)

2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)
2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)

2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)
2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)
2𝐷𝐶𝑜𝑛𝑣(5 ∗ 5)

𝑀𝑎𝑥 − 𝑝𝑜𝑜𝑙𝑖𝑛𝑔(2 ∗ 2)
Flaten
Dense
Dropout (0.5)
Dense
Sofmax

Fig. 8. Variation of loss value for training and testing Data in DS2 data set.

Fig. 9. Variation of accuracy value for training and testing data in DS3 data set.

Fig. 10. Variation of loss value for training and testing data in DS3 data set.

uncorrelated to previous components, which maximizes the variance
of the samples when projected onto the component (Ringnér, 2008).
The scatter diagrams are used to determine the relationship between
different molecules within a class. This relationship was based on
the new individual learning representations of molecules, which were
reduced to 3D structures using the PCA technique to represent their
characteristics.

method. Thereafter, for each reference structure in a data set, a similarity measurement is carried out in order to sort the database based on
a decreasing order of similarity with the reference structure to identify
the active molecules. Finally, the average number of active molecules
in 1% and 5% cut-offs is determined for a specific class by averaging
ten different results.
The experimental results obtained by our method and other recognized similarity methods are presented in Tables 10–15. These results
are calculated on the MDDR dataset (DS1, DS2, and DS3) and are
organized as follows: Each data set has two tables for the 1% and 5%
cut-offs. Thereafter, the first columns of each table contain the indexes

The virtual screening search process is simulated using ten reference
structures of each activity class for all data sets. Subsequently, the
selected references were unified and implemented in our DCNNLB
9

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 11. Graphical representation of 3D dispersion of all compounds of the eleven activity classes for DS1 data set based on new learning representations using PCA.

Fig. 12. Graphical representation of 3D dispersion of all compounds of the ten activity classes for DS2 data set based on new learning representations using PCA.

Fig. 13. Graphical representation of 3D dispersion of all compounds of the ten activity classes for DS3 data set based on new learning representations using PCA.
Table 6
The obtained results of testing model using different kernel size and 2DConv layers
numbers for all MDDR datasets.
Model

A

B

C

D

E

F

2DConv number
Kernel size

1
3

2
3

3
3

1
5

2
5

3
5

Accuracy for DS1
Accuracy for DS2
Accuracy for DS3

0.9914
0.9873
0.9774

0.9984
0.9904
0.9946

0.9989
0.9915
0.9943

0.9814
0.9856
0.9793

0.9977
0.9897
0.9938

0.9976
0.9889
0.9918

Mean

0,9854

0,9945

0,9949

0,9821

0,9937

0,9928

of the activity classes of each data set, while the first row, contains
the names of the similarity methods that we quoted to compare the
performance of our proposed method. These methods are (in order):
Bayesian inference Network (BIN) (Abdo & Salim, 2009), Tanimoto
method (TAN), Standard Quantum-Based method (SQB) (Al-Dabbagh
et al., 2015), Feature Selection using Genetic Algorithm for Similarity
Searching in LBVS method (FSGASS) (Berrhail & Belhadef, 2020),
QPRP-Complex method (QPRP-C) (Al-Dabbagh et al., 2017), and the
last one is our proposed DCNNLB method. In addition, each row for
each table shows the calculated recall for the top 1% and top 5% of
the activity class. The best recall rate for each row is then shaded. In
all tables, the average rows represent the recall average calculated by
taking into account all activity classes (the best mean is shown in bold).

Fig. 14. Comparison of the performance values (recall) of similarity methods for DS1
dataset in Top 1% using ANOVA.

The shaded cells in the rows represent the total amount of shaded cells
for each method across all activity classes.
According to the results obtained of the global recall values for the
MDDR data sets, which are presented in Tables 10–15. Our proposed
DCNNLB method which is based on Deep Convolution Neural Network
model is superior to the BIN, TAN, SQB, FSGASS and QPRP-C methods
in terms of performance (Overall recall) in all data sets; except for the
DS2 dataset in Top 1%. However, we noticed that the FSGASS method
shows a good overall recall in DS2 at Top 1%.
10

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.
Table 7
CNN configuration details for varying the number of convolutional layers and kernel sizes.
CNN model configuration
G

H

I

J

K

L

M

One layer

Two layers

Three layers

Three layers

Four layers

Five layers

Six layers

2DConv
Max-pool

2DConv
2DConv
Max-pool

2DConv
Max-pool
2DConv
Max-pool
2DConv
Max-pool

2DConv
2DConv
2DConv
Max-pool

2DConv
2DConv
Max-pool
2DConv
Max-pool

2DConv
2DConv
Max-pool
2DConv
2DConv
Max-pool
2DConv
Max-pool

2DConv
2DConv
2DConv
Max-pool
2DConv
2DConv
2DConv
Max-pool

Flaten
Dense
Dropout (0.5)
Dense
Sofmax

Table 8
The obtained results of testing model using different CNN model topology for all MDDR
datasets.
Model

G

H

I

J

K

L

Table 11
The obtained results of recall values in top 5% for the DS1 dataset.
Activity index

Conv2D number

1

2

3

3

4

6

8

DS1 accuracy
DS2 accuracy
DS3 accuracy

0.9362
0.9649
0.9401

0.9553
0.9843
0.9918

0.8654
0.9562
0.7232

0.9943
0.9834
0.9913

0.9375
0.9730
0.9058

0.9143
0.9514
0.6888

0.8988
0.9660
0.8857

Mean

0,9471

0,9771

0,8483

0,9897

0,9388

0,8515

0,9168

Layer
index

Layer type

Number of
filters

Filter
size

Parameters

Out shape

1
2
3
4
5
6
7
8
9
10

Input Layer
2D Convolution
2D Convolution
2D Convolution
Max-Pooling
Flatten
Dense
Dropout
Dense
Dense (Softmax)

–
16
16
16
–
–
–
–
–
–

–
3∗3
3∗3
3∗3
2∗2
–
–
–
–
–

0
160
2320
2320
0
0
2 705 000
0
380 380
4191

(32, 32, 1)
(30, 30, 16)
(28, 28, 16)
(26, 26, 16)
(13, 13, 16)
2704
1000
380
380
11

31 420
71 523
37 110
31 432
42 731
6233
6245
7701
6235
78 374
78 331

TAN

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

70.79
31.37
25.79
43.14
18.16
13.34
6.46
12.08
11.7
14.44
5.91

77.92
29.89
23.55
43.57
24.82
16.41
7.61
12.42
14.63
16.82
7.93

80,80
59,92
39,65
45,12
48,60
35,25
21,64
19,95
21,55
29,62
13,63

Top 1 %
31 420
71 523
37 110
31 432
42 731
6233
6245
7701
6235
78 374
78 331

74.08
28.26
26.05
29.23
21.68
14.06
6.31
11.45
10.84
14.25
6.03

69.75
27.68
23.21
40.32
17.02
14.57
6.3
10.48
10.99
12.74
6.78

73.73
26.84
24.73
36.66
21.17
12.49
6.03
11.35
10.15
13.08
5.92

Mean

22.93

21.8

22.01

23.02

25.05

37,79

Shaded cells

0

0

0

0

0

11

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

83.42
52.96
50.75
82
29.44
25.57
15.93
29.85
26.82
22.25
12.94

87.15
60.92
44.54
86.72
32.1
25.79
19.78
30.84
25.33
22.13
11.93

90,12
86,31
75,95
97,17
72,32
67,22
52,09
52,51
50,41
46,16
37,64

87.61
52.72
48.2
77.57
26.63
23.49
14.86
27.79
23.78
20.2
11.8

86.28
54.55
45.57
79.13
26.16
26.4
14.21
26.08
24
19.43
13.02

87.22
48.7
45.62
70.44
19.35
21.04
13.63
21.85
19.13
20.55
13.1

Mean

37.69

37.71

34.05

39.25

40.7

66,17

Shaded cells

0

0

0

0

0

11

explains their superiority over the other methods. In the Top 5%,
our method was able to obtain the best overall recall despite having
only good average recall in three out of ten classes. This explains their
efficiency to search and find active molecules, especially in the first
activity class, where it recorded a rate of superiority of 14.73% compared to the FSGASS method. Moreover, in DS3 dataset, our method
performed better than the other method in nine activity classes for the
Top 1% and Top 5%.
Therefore, the calculated average recall of our proposed method
DCNNLB for all MDDR datasets is better than the BIN, TAN, SQB
FSGASS and QPRP-C methods.
Furthermore, we used the ANOVA test (Analysis of Variance) technique to compare and evaluate the performance of the research process
of our proposed DCNNDL method with other methods such as BIN,
TAN, SQB, FSGASS and QPRP-Complex method. The analysis of variance or ANOVA test is considered a technique more widely used in the
literature to compare means or variance when there are more than two
values to compare. Thus, in this work, ANOVA test was used to analyze
and compare the screening effectiveness of the similarity methods using
all MDDR datasets. For all activity classes of MDDR data bases the
average recall values were considered as rankings for the similarity
methods. Therefore, it is possible to accord an overall ranking to the
similarity methods. The comparison of the screening recall values of
all methods of similarity in the Top 1% and Top 5% for all MDDR data
sets are shown in Figs. 14–19.
The comparison of the screening recall values obtained in data set
DS1 for the Top 1%, which is shown in Fig. 14, showed that our
proposed DCNNLB method has the best recall value with a median of:
36.52 and a 𝑝-value equal to 0.25. Therefore, the ranking of the six similarity methods were: 𝐷𝐶𝑁𝑁𝐿𝐵 > 𝑄𝑃 𝑅𝑃 − 𝐶(20, 18) > 𝐵𝐼𝑁(17.96) >

Table 10
The obtained results of recall values in top 1% for the DS1 dataset.
BIN

TAN

Top 5%

Table 9
The details of the configuration of our proposed DCNN model.

Activity index

BIN

M

The DS1 data set showed good results for our proposed DCCNLB
method, given the eleven out of eleven classes in cut-off 1% and 5%.
Therefore, in the DS2 we find that our method achieved good average
recall results in four out of ten classes for the Top 1%. While the
FSGASS method obtains a higher average recall in two activity classes,
with a better overall average recall in the Top 1%; this
11

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.
Table 12
The obtained results of recall values in top 1% for the DS2 dataset.
Activity index

BIN

TAN

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

71.45
96.34
73.49
94.50
86.62
74.74
69.73
92.22
81.21
96.22

73.41
96.75
85.25
83.91
88.61
74.02
65.42
81.28
81.58
97.93

73,86
83,85
65,97
92,07
87,32
76,65
82,39
87,94
75,41
98,46

Top 1%
7707
7708
31 420
42 710
64 100
64 200
64 220
64 500
64 350
75 755

72.18
96.00
79.82
74.27
88.43
68.18
68.32
81.2
81.89
97.06

72.08
97.18
73.57
80.72
86.34
63.29
68.00
73.73
82.09
97.80

72.09
95.68
78.56
76.82
87.8
70.18
67.58
79.2
81.68
98.02

Mean

80.73

79.53

80.76

83.65

82.81

82,39

Shaded cells

0

1

0

2

3

4

Table 13
The obtained results of recall values in top 5% for the DS2 dataset.
Activity index

BIN

TAN

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

75.80
100
92.14
98.38
99.76
99.11
95.84
99.29
99.41
97.80

75.37
100
94.76
93.27
99
98.84
93.79
96.64
91.48
98.28

90,53
95,51
94,03
98,29
98,62
91,90
99,64
98,89
99,28
99,67

Top 5%
7707
7708
31 420
42 710
64 100
64 200
64 220
64 500
64 350
75 755

73.81
98.61
93.46
92.55
98.22
97.5
91.32
94.96
91.47
97.33

74.88
99.94
94.11
91.35
99.49
96.01
90.17
89.44
89.97
98.24

74.22
100
95.24
93
98.94
98.93
90.9
92.72
93.75
95.39

Mean

92.92

92.36

93.31

95.75

94.14

96,64

Shaded cells

0

0

1

6

0

3

Table 14
The obtained results of recall values in top 1% for the DS3 dataset.
Activity index

BIN

TAN

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

17.50
8.57
12.06
28.58
10.55
7.70
24.78
5.36
8.85
16.41

14.66
8.37
9.15
20.48
8.25
8.87
21.07
7.28
9.8
12.98

74,71
28,08
20,95
36,89
41,53
52,53
26,38
27,56
17,42
11,52

Top 1%
9249
12 455
12 464
31 281
43 210
71 522
75 721
78 331
78 348
78 351

12.69
7.7
7.9
19.24
7.32
6.61
21.46
6.5
8.7
13.23

16.12
8.70
10.16
23.68
9.25
5.47
26.34
9.89
9.37
15.08

10.99
7.03
6.92
18.67
6.83
6.57
20.38
6.16
8.99
12.5

Mean

11.13

13.40

10.50

14.04

12.09

33,76

Shaded cells

0

0

0

1

0

9

Table 15
The obtained results of recall values in top 5% for the DS3 dataset.
Activity index

BIN

TAN

SQB

FSAGSS

QPRP-Complex

Our DCNNLB

23.69
10.34
21.53
27.52
15.38
15.02
29.8
11.45
22.81
13.66

98,30
58,84
47,74
85,47
73,75
86,54
50,00
52,08
50,18
16,49

Top 5%
9249
12 455
12 464
31 281
43 210
71 522
75 721
78 331
78 348
78 351

20.1
10.73
19.7
29.52
14.31
14.41
31.8
12.72
20.8
14.54

28.16
12.56
19.17
36.04
16.94
10.11
35.88
17.04
20.73
17.16

17.8
11.42
16.79
29.05
14.12
13.82
30.61
11.97
21.14
13.3

31.34
13.26
24.69
42.55
19.81
13.09
35.20
11.93
18.91
18.13

Mean

18.86

21.37

18.0

22.89

19.12

61,94

Shaded cells

0

0

0

1

0

9

12

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 15. Comparison of the performance values (recall) of similarity methods for DS1 dataset in Top 5% using ANOVA.

Fig. 16. Comparison of the performance values (recall) of similarity methods for DS2 dataset in Top 1% using ANOVA.

Fig. 17. Comparison of the performance values (recall) of similarity methods for DS2 dataset in Top 5% using ANOVA.

𝑄𝑃 𝑅𝑃 − 𝐶(30.84) > 𝐹 𝑆𝐺𝐴𝑆𝑆(29.44) > 𝐵𝐼𝑁(26.63) > 𝑇 𝐴𝑁(26.16) >
𝑆𝑄𝐵(21.04).
In Fig. 16, we present the results obtained by the Anova test for
the DS2 data set in the Top 1%. A median value of 83.91 for the
FSGASS method shows a superior in performance compared to the other
methods. Thus, the overall ranking of the six methods is as follows:
𝐹 𝑆𝐺𝐴𝑆𝑆 > 𝐷𝐶𝑁𝑁𝐿𝐵(83.12) > 𝑄𝑃 𝑅𝑃 − 𝐶(82.74) > 𝐵𝐼𝑁(80.51) >
𝑆𝑄𝐵(78.88) > 𝑇 𝐴𝑁(77.22). In addition, the results obtained from the
ANOVA test in DS2 for Top 5% (Fig. 17) showed that the proposed
DCNNLB method has a higher mean recall value than all other methods
(96,64), with a median of 98.75, 98.74 for FSGASS, 95.7 for QPRP-C,
94.49 for SQB, 94.21 for BIN and 92.73 for TAN. Thus, the overall
ranking of the six similarity methods is as follows: 𝐷𝐶𝑁𝑁𝐿𝐵 >
𝐹 𝑆𝐺𝐴𝑆𝑆 > 𝑄𝑃 𝑅𝑃 − 𝐶 > 𝑆𝑄𝐵 > 𝐵𝐼𝑁 > 𝑇 𝐴𝑁.
The Fig. 18 shows the screening recall results obtained by ANOVA
test for MDDR DS3 at Top 1%. These results showed a superior performance of our method, with a 𝑝-value equal to 3.75 *10-̂6 that means a
significant difference in performance between similarity methods. The
median values for the DCNNLB, FSGASS, QPRP-C, TAN, BIN, and SQB
methods were: 27.82, 11.30, 9.47, 10.02, 8.3, and 8.01, respectively.
Thus, the overall ranking of similarity methods is as follows: DCNNLB
> FSGASS > QPRP-C > TAN > BIN > SQB. The comparison shown

Fig. 18. Comparison of the performance values (recall) of similarity methods for DS3
dataset in Top 1% using ANOVA.

𝑆𝑄𝐵(17.12) > 𝐹 𝑆𝐺𝐴𝑆𝑆(16.3) > 𝑇 𝐴𝑁(15.79). The results obtained in
the 5% cut-off for the Anova test analysis for the DS1 data set are
shown in Fig. 15. This latter indicates, that our method gives the best
mean recall value with the median: 67.22 and a 𝑝-value of 0.041 which
shows a significant difference in the performance of our method. Thus,
the overall ranking of similarity methods is as follows: 𝐷𝐶𝑁𝑁𝐿𝐵 >
13

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Fig. 19. Comparison of the performance values (recall) of similarity methods for DS3 dataset in Top 5% using ANOVA.

List of acronyms

in Fig. 19 of the screening recall values obtained in DS3, at top
5% demonstrated, that the DCNNLB method presents a superiority in
performance with a 𝑝-value equal to 5.8* 10-̂11 (a significant difference
in performance) and the median values were found at : 55.46, 19.36,
18.45, 18.16, 15.45, 17.12 for the CNNLB, FSGASS, QPRP-C, TAN, SQB,
and BIN, respectively. As a result, the overall ranking of similarity
methods is as follows: 𝐷𝐶𝑁𝑁𝐿𝐵 > 𝐹 𝑆𝐺𝐴𝑆𝑆 > 𝑄𝑃 𝑅𝑃 − 𝐶 > 𝑇 𝐴𝑁 >
𝐵𝐼𝑁 > 𝑆𝑄𝐵.

## 6. Conclusion

Abbreviation

Signification

ANN
ANOVA
BIN
CNN
DCNN
DCNNLB

: Artificial Neural Networks.
: ANalyse Of Variance.
: Bayesian inference Network.
: Convolutional Neural Network.
: Deep Convolutional Neural Network.
: Deep Convolutional Neural Network for
Ligand-Based Virtual screening.
: Deep Learning.
: Extended Connectivity Fingerprint Counts.
: Feature Selection using Genetic Algorithm for
Similarity Searching in LBVS.
: Swarm intelligence.
: K-Nearest Neighbors.
: MDL Drug Data Report.
: Machine Learning.
: Principal Component Analysis.
: Quantum Probability Ranking Principle.
: Quantitative Structure-Activity Relationship.
: Structure-Based Virtual Screening.
: Standard Quantum-Based.
: Support Vector Machines.
: Tanimoto.
: Virtual Screening.

DL
ECPC
FSGASS

In this work, a Deep Convolutional Neural Network for LBVS
method is proposed to enhance the performance of similarity searching
in ligand-based virtual screening process. In our first main contribution, we have succeeded in creating and designing a DCNN model for
improving the effectiveness of the screening process and the prediction
of the biological activity of molecules. This model receives as input
the Com2Mat representation of the chemical compounds which are
obtained by re-representing the compounds from their 2D Fingerprint
descriptors. Furthermore, we have proposed several network topologies in order to find a learning model that can initially provide better
performance. The proposed model is trained on all compounds of
all activity classes of the MDDR datasets. Thus, it has presented a
mean accuracy of 0.98 for all three MDDR Datasets. In the second
contribution, a new learning representation is generated in order to
better represent chemical compounds. This representation is based on
the extraction of the automatic features learning from the weights of
our proposed model.

IS
K-NN
MDDR
ML
PCA
QPRP
QSAR
SBVS
SQB
SVM
TAN
VS

CRediT authorship contribution statement

The screening experiments are performed on the MDDR datasets
to test and assess our proposed DCNNLB method. An Enhancement in
effectiveness of LBVS process is obtained by application of our method,
which achieved good average recall results notably in Top 5% in all
MDDR datasets (i.e 66.17 for MDDR-DS1, 96.64 for MDDR-DS2, and
61.94 for MDDR-DS3). Therefore, this improvement in performance
allows to increase the number of active molecules that will contribute
in the design of new drug candidates notably in the lead optimization
phase of the drug design process. Thereafter, ANOVA test is used to
compare the effectiveness of our proposed DCNNLB method with the
conventional method (TAN) and the other methods such as : BIN, SQB,
QPRP-C and FSGASS. The results obtained and the overall ranking of
the compared similarity methods given for all datasets in the Top 1%
and Top 5% are evidence of the reliability of our proposals. Thus, the
proposed DCNNLB approach performs better than the TAN, BIN, SQB,
FSGASS and QPRP-C method.

Fouaz Berrhail: Conceptualization, Designed the model and the
computational framework, Methodology, Datasets preparation and analysis, Software, Carried out the experiment, analysis of the results, Writing – original draft, Writing and editing, Visualization, Investigation,
validation. Hacene Belhadef: Conceptualization, Designed the model,
Supervision. Mohammed Haddad: Conceptualization, Designed the
model, Analysis of the results, Supervision.
Declaration of competing interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.
## References
Abdo, A., & Salim, N. (2009). Similarity-based virtual screening with a bayesian
inference network. ChemMedChem: Chemistry Enabling Drug Discovery, 4, 210–218.
http://dx.doi.org/10.1002/cmdc.200800290.
Abdo, A., & Salim, N. (2011). New fragment weighting scheme for the bayesian
inference network in ligand-based virtual screening. Journal of Chemical Information
and Modeling, 51, 25–32. http://dx.doi.org/10.1021/ci100232.

Finally, in the future work, we hope to apply the techniques of
merging several deep CNNs and LSTMs (Long short-term memory) to
enhance LBVS process and to optimize the learning molecular representation.
14

Expert Systems With Applications 203 (2022) 117287

F. Berrhail et al.

Karnachi, P. S., & Brown, F. K. (2004). Practical approaches to efficient screening:
information-rich screening protocol. Journal of Biomolecular Screening, 9, 678–686.
http://dx.doi.org/10.1177/1087057104269570.
Kauffman, G. W., & Jurs, P. C. (2001). Qsar and k-nearest neighbor classification analysis of selective cyclooxygenase-2 inhibitors using topologically-based numerical
descriptors. Journal of Chemical Information and Computer Sciences, 41, 1553–1560.
http://dx.doi.org/10.1021/ci010073h.
keras (2022). Keras and tensorflow. available online: https://keras.io/api/models
(accessed on 21 march 2022). https://keras.io/api/models.
Konovalov, D. A., Coomans, D., Deconinck, E., & Vander Heyden, Y. (2007). Benchmarking of qsar models for blood–brain barrier permeation. Journal of Chemical
Information and Modeling, 47, 1648–1656. http://dx.doi.org/10.1021/ci700100f.
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). Imagenet classification with
deep convolutional neural networks. Communications of the ACM, 60, 84–90. http:
//dx.doi.org/10.1145/3065386.
Lo, Y.-C., Rensi, S. E., Torng, W., & Altman, R. B. (2018). Machine learning in
chemoinformatics and drug discovery. Drug Discovery Today, 23, 1538–1546. http:
//dx.doi.org/10.1016/j.drudis.2018.05.010.
Ragoza, M., Hochuli, J., Idrobo, E., Sunseri, J., & Koes, D. R. (2017). Protein–ligand
scoring with convolutional neural networks. Journal of Chemical Information and
Modeling, 57, 942–957. http://dx.doi.org/10.1021/acs.jcim.6b00740.
Rawlins, M. D. (2004). Cutting the cost of drug development? Nature Reviews Drug
Discovery, 3, 360–364. http://dx.doi.org/10.1038/nrd1347.
Ringnér, M. (2008). What is principal component analysis? Nature biotechnology, 26,
303–304. http://dx.doi.org/10.1038/nbt0308-303.
Su, B., & Lu, S. (2017). Accurate recognition of words in scenes without character
segmentation using recurrent neural network. Pattern Recognition, 63, 397–405.
http://dx.doi.org/10.1016/j.patcog.2016.10.016.
Sun, Y., Liu, Y., Wang, G., & Zhang, H. (2017). Deep learning for plant identification
in natural environment. Computational Intelligence and Neuroscience, 2017. http:
//dx.doi.org/10.1155/2017/7361042.
Wallach, I., Dzamba, M., & Heifets, A. (2015). Atomnet: a deep convolutional neural
network for bioactivity prediction in structure-based drug discovery. arXiv preprint
arXiv:1510.02855. doi:https://arxiv.org/abs/1510.02855.
Wang, H., & Raj, B. (2017). On the origin of deep learning. arXiv preprint arXiv:
1702.07800. doi:https://arxiv.org/abs/1702.07800.
Whittle, M., Gillet, V. J., Willett, P., & Loesel, J. (2006). Analysis of data fusion methods
in virtual screening: similarity and group fusion. Journal of Chemical Information and
Modeling, 46, 2206–2219. http://dx.doi.org/10.1021/ci0496144.
Whittle, M., Willett, P., Klaffke, W., & van Noort, P. (2003). Evaluation of similarity
measures for searching the dictionary of natural products database. Journal of
Chemical Information and Computer Sciences, 43, 449–457. http://dx.doi.org/10.
1021/ci025591m.
Willett, P. (2013). Combination of similarity rankings using data fusion. Journal of
Chemical Information and Modeling, 53, 1–10. http://dx.doi.org/10.1021/ci300547g.
Willett, P., Barnard, J. M., & Downs, G. M. (1998). Chemical similarity searching.
Journal of Chemical Information and Computer Sciences, 38, 983–996. http://dx.doi.
org/10.1021/ci9800211.
Wu, F., Jing, X.-Y., Feng, Y., Ji, Y.-m., & Wang, R. (2020). Spectrum-aware discriminative deep feature learning for multi-spectral face recognition. Pattern Recognition,
111, Article 107632. http://dx.doi.org/10.1016/j.patcog.2020.107632.
Zurada, J. M. (1992). Introduction to artificial neural systems, vol. 8. West St. Paul.

Ahmed, A., Abdo, A., & Salim, N. (2012a.). An enhancement of bayesian inference
network for ligand-based virtual screening using minifingerprints. In Fourth international conference on machine vision (ICMV 2011): computer vision and image analysis;
pattern recognition and basic technologies, vol. 8350 (p. 83502U). International Society
for Optics and Photonics, http://dx.doi.org/10.1117/12.920338.
Ahmed, A., Abdo, A., & Salim, N. (2012b). Ligand-based virtual screening using
bayesian inference network and reweighted fragments. The Scientific World Journal,
2012, 1–7. http://dx.doi.org/10.1100/2012/410914.
Al-Dabbagh, M., Salim, N., Himmat, M., Ahmed, A., & Saeed, F. (2015). A quantumbased similarity method in virtual screening. Molecules, 20, 18107–18127. http:
//dx.doi.org/10.3390/molecules201018107.
Al-Dabbagh, M. M., Salim, N., Himmat, M., Ahmed, A., & Saeed, F. (2017). Quantum
probability ranking principle for ligand-based virtual screening. Journal of ComputerAided Molecular Design, 31, 365–378. http://dx.doi.org/10.1007/s10822-016-00034.
Atto, A. M., Benoît, A., & Lambert, P. (2020). Timed-image based deep learning
for action recognition in video sequences. Pattern Recognition, Article 107353.
http://dx.doi.org/10.1016/j.patcog.2020.107353.
Bajusz, D., Rácz, A., & Héberger, K. (2015). Why is tanimoto index an appropriate
choice for fingerprint-based similarity calculations? Journal of Cheminformatics, 7,
20. http://dx.doi.org/10.1186/s13321-015-0069-3.
Benedict, J. (2021). Report, mdl drug data: Sci tegic accelrys inc. the mdl drug data
report (mddr). available online: http://accelrys.com (accessed on 21 april 2021).
URL: http://accelrys.com.
Berrhail, F., & Belhadef, H. (2020). Genetic algorithm-based feature selection approach for enhancing the effectiveness of similarity searching in ligand-based
virtual screening. Current Bioinformatics, 15, 431–444. http://dx.doi.org/10.2174/
1574893614666191119123935.
Berrhail, F., Belhadef, H., Hentabli, H., & Saeed, F. (2017). Molecular similarity
searching with different similarity coefficients and different molecular descriptors.
In International conference of reliable information and communication technology (pp.
39–47). Springer, http://dx.doi.org/10.1007/978-3-319-59427-9_5.
Biovia, P. (2021). Pipeline pilot software : Scitegic accelrys inc. https://www.3dsbiovia.
com/products/collaborative-science/biovia-pipeline-pilot/ (accessed on 21 april
2021), https://www.3dsbiovia.com/products/collaborative-science/biovia-pipelinepilot/.
Brezočnik, L., Fister, I., & Podgorelec, V. (2018). Swarm intelligence algorithms for
feature selection: a review. Applied Sciences, 8(1521), http://dx.doi.org/10.3390/
app8091521.
Cereto-Massagué, A., Ojeda, M. J., Valls, C., Mulero, M., Garcia-Vallvé, S., & Pujadas, G.
(2015). Molecular fingerprint similarity search in virtual screening. Methods, 71,
58–63. http://dx.doi.org/10.1016/j.ymeth.2014.08.005.
Dahl, G. E., Sainath, T. N., & Hinton, G. E. (2013). Improving deep neural networks for
lvcsr using rectified linear units and dropout. In 2013 IEEE international conference
on acoustics, speech and signal processing (pp. 8609–8613). IEEE, http://dx.doi.org/
10.1109/ICASSP.2013.6639346.
Fouaz, B., Hacene, B., Hamza, H., & Saeed, F. (2019). Similarity searching in
ligand-based virtual screening using different fingerprints and different similarity
coefficients. International Journal of Intelligent Systems Technologies and Applications,
18, 405–425. http://dx.doi.org/10.1504/IJISTA.2019.100809.
Johnson, M. A., & Maggiora, G. M. (1990). Concepts and applications of molecular
similarity. Wiley.
Jorissen, R. N., & Gilson, M. K. (2005). Virtual screening of molecular databases using a
support vector machine. Journal of Chemical Information and Modeling, 45, 549–561.
http://dx.doi.org/10.1021/ci049641u.

15


