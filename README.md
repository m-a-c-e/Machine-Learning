# ML 7641: Team 21
## Project Title: Multi-Modal Sarcasm Detection

### Introduction:
Sarcasm detection is a task under sentiment analysis that involves recognizing sarcastic utterances whose perlocutionary effect requires a great understanding of the conversational context, the utterance, and some basic knowledge to perceive the whole conversation. The ironic and metaphorical nature of sarcasm poses a challenge for computer models that try to analyze the sentiment of the conversation.

### Problem Definition:
Several NLP models have tried to detect sarcasm in text statements using annotation <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf">[1]</a> or context incongruity [2]. Though these models have managed to achieve good accuracy by including context and embeddings of personality features [3], the existing models do not focus on sarcasm detection in conversation. Past work in sarcasm detection using speech is based on identifying prosodal and spectral features that corresponds to sarcastic comments [4]. Visual markers to understand sarcasm have been rarely studied. Thus, in our project, we aim to detect sarcasm in video segments of dialogues by extracting visual features, and combining it with the corresponding textual and audio cues to derive additional contextual information [5] to identify sarcastic behavior. The motivation for sarcasm detection is to improve Conversational AI’s ability to perceive human interaction, Opinion mining, Marketing research, and Information categorization. 

### Dataset Description:
MUStARD(multimodal video corpus) [6] consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which offers more details about the situation in which the utterance takes place. The dataset consists of 345 sarcastic videos and 6,020 non-sarcastic videos. The dataset is compiled from popular TV shows.

<!---
**Features of a Datapoint:**: 
- Video Sequence for Utterance
- Video Sequence for Context
- Transcripts of the Videos
- Speaker Name and Context character names
- Label (Sarcastic or not)


![sample_datapoint.jpg](./Images/sample_datapoint.PNG) 
|:--:| 
| **Utterance and Context Video sequences: Text and Audio-visual components (Credit: [6])** |

![sample_json.jpg](./Images/sample_json.PNG) 
|:--:| 
| **Example of a JSON file that contains all information about the video sequences for a datapoint (Credit: [6])** |
--->


<p align="center">
<img src="./Images/sample_datapoint.PNG" style="border: 1px solid black" >
<figcaption align="middle">Utterance and Context Video sequences: Text and Audio-visual components (Credit: [6])</figcaption>
</p>


<p align="center">
<img src="./Images/sample_json.PNG" style="border: 1px solid black" >
<figcaption align="middle">Example of a JSON file that contains all information about the video sequences for a datapoint (Credit: [6])</figcaption>
</p>


### Methods:
A feature vector will be generated for each datapoint that will be a combination of its audio, video, and text.
- **Video**: CNNs like VGG, ImageNet, ResNet, etc. 
- **Audio**:  RNN models like Vanilla RNN and Librosa library [6] (MFCC, melspectogram, spectral centroid). 
- **Text**: RNN models like LSTM and Transformers like BERT[1]. <br>


The feature vectors of the 3 modalities will be combined and given as input to the supervised or unsupervised Machine Learning models. 

**Supervised methods:**
* Logistic Regression
* Naive Bayes
* Support Vector Machines
* Deep Neural Networks

**Unsupervised Methods:**
* K-Mean Clustering
* Gaussian Mixture Model

<!---
![data_flowchart.jpg](./Images/data_flowchart.png) 
|:--:| 
| **Feature Extraction and Data Analysis Flowchart** |
--->

<p align="center">
<img src="./Images/data_flowchart.png" style="border: 1px solid black" >
<figcaption align="middle">Feature Extraction and Data Analysis Flowchart</figcaption>
</p>


### Potential Results and Discussion:
The goal of this project is to classify if a video is sarcastic or not given its audio, video and captions. The classifier model will output probability of a datapoint being sarcastic. The clustering models will distinguish between sarcastic and non-sarcastic content. Since the dataset is imbalanced, metrics like Precision, Recall, F1-Score, ROC-AUC score will be used to evaluate the Supervised Learning models. The clustering algorithms will be evaluated using Silhouette Coefficient.
Furthermore, the interpretability of the model will also be analyzed using AI Explainability tools.

### References:
[1] Lydia Xu, Vera Xu. "Project Report: Sarcasm Detection" [Link](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf) <br>
[2] Joshi, Aditya, Vinita Sharma, and Pushpak Bhattacharyya. "Harnessing context incongruity for sarcasm detection." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2015. [Link](https://aclanthology.org/P15-2124.pdf) <br>
[3] Poria, Soujanya, et al. "A deeper look into sarcastic tweets using deep convolutional neural networks." arXiv preprint arXiv:1610.08815 (2016). [Link](https://arxiv.org/pdf/1610.08815.pdf) <br>
[4] Tepperman, Joseph, David Traum, and Shrikanth Narayanan. "" Yeah right": sarcasm recognition for spoken dialogue systems." Ninth international conference on spoken language processing. 2006. [Link](http://www1.cs.columbia.edu/~julia/papers/teppermanetal06.pdf) <br>
[5] Byron C. Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak. 2014. "Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)". In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 512–516, Baltimore, Maryland. Association for Computational Linguistics. [Link](https://aclanthology.org/P14-2084) <br>
[6] Castro, Santiago, et al. "Towards multimodal sarcasm detection (an _obviously_ perfect paper)." arXiv preprint arXiv:1906.01815 (2019). [Link](https://arxiv.org/pdf/1906.01815.pdf)

