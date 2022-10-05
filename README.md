# ML 7641: Team 21
## Project Title: Multi-Modal Sarcasm Detection

### Introduction:
Sarcasm detection is a task under sentiment analysis that involves recognizing sarcastic utterances whose perlocutionary effect requires a great understanding of the conversational context, the utterance, and some basic knowledge to perceive the whole conversation. The ironic and metaphorical nature of sarcasm poses a challenge for computer models that try to analyze the sentiment of the conversation.

### Problem Definition:
Several NLP models have tried to detect sarcasm in text statements using annotation or context incongruity. Though these models have managed to achieve good accuracy by including context and embeddings of personality traits[1], the existing models do not focus on sarcasm detection in conversation. Past work in sarcasm detection using speech is based on intonation that corresponds to sarcastic behavior[2]. Visual markers to understand sarcasm have been rarely studied. Thus, in our project, we aim to detect sarcasm in video segments of dialogues by extracting visual features, and combining it with the corresponding textual and audio cues to derive additional contextual information to identify sarcastic behavior. The motivation for sarcasm detection is to improve Conversational AI’s ability to perceive human interaction, Opinion mining, Marketing research, and Information categorization [3]. 

### Dataset Description:
MUStARD(multimodal video corpus) consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which offers more details about the situation in which the utterance takes place. The dataset consists of 345 sarcastic videos and 6,020 non-sarcastic videos. The dataset is compiled from popular TV shows.


![sample_datapoint.jpg](./Images/sample_datapoint.PNG) 
|:--:| 
| **Utterance and Context Video sequences: Text and Audio-visual components** |


![sample_json.jpg](./Images/sample_json.PNG) 
|:--:| 
| **Example of a JSON file that contains all information about the video sequences for a datapoint** |

### Methods:
A feature vector will be generated for each datapoint that will be a combination of its audio, video, and text. The feature vector is generated as follows:
- Video: CNNs like ImageNet, ResNet, etc. 
- Audio:  RNN models like Vanilla RNN and Librosa library (MFCC, melspectogram, spectral centroid). 
- Text: RNN models like LSTM and Transformers like BERT. [4]
The feature vectors of the 3 modalities will be combined and given as input to the supervised or unsupervised Machine Learning models. 

**Supervised methods:**
* Logistic Regression
* Naive Bayes
* Support Vector Machines
* Deep Neural Networks

**Unsupervised Methods:**
* K-Mean Clustering
* Gaussian Mixture Model

![data_flowchart.jpg](./Images/data_flowchart.png) 
|:--:| 
| **Feature Extraction and Data Analysis Flowchart** |

### Potential Results and Discussion:
The goal of this project is to classify if a video is sarcastic or not given its audio, video and captions. The classifier model will output probability of a datapoint being sarcastic. The clustering models will distinguish between sarcastic and non-sarcastic content. Since the dataset is imbalanced, metrics like Precision, Recall, F1-Score, ROC-AUC score will be used to evaluate the Supervised Learning models. The clustering algorithms will be evaluated using Silhouette Coefficient.
Furthermore, the interpretability of the model will also be analyzed using AI Explainability tools.
