# ML 7641: Team 21

# Project Proposal
## Project Title: Multi-Modal Sarcasm Detection

### Introduction:
Sarcasm detection is a task under sentiment analysis that involves recognizing sarcastic utterances whose perlocutionary effect requires a great understanding of the conversational context, the utterance, and some basic knowledge to perceive the whole conversation. The ironic and metaphorical nature of sarcasm poses a challenge for computer models that try to analyze the sentiment of the conversation.

### Problem Definition:
Several NLP models have tried to detect sarcasm in text statements using context incongruity <a href="https://aclanthology.org/P15-2124.pdf">[1]</a> or Deep Learning NLP <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf">[2]</a>. Though these models have managed to achieve good accuracy by including context and embeddings of personality features <a href="https://arxiv.org/pdf/1610.08815.pdf">[3]</a>, the existing models do not focus on sarcasm detection in conversation. Past work in sarcasm detection using speech is based on identifying prosodal and spectral features that corresponds to sarcastic comments <a href="http://www1.cs.columbia.edu/~julia/papers/teppermanetal06.pdf">[4]</a>. Visual markers to understand sarcasm have been rarely studied. Thus, in our project, we aim to detect sarcasm in video segments of dialogues by extracting visual features, and combining it with the corresponding textual and audio cues to derive additional contextual information <a href="https://aclanthology.org/P14-2084">[5]</a> to identify sarcastic behavior. The motivation for sarcasm detection is to improve Conversational AI’s ability to perceive human interaction, Opinion mining, Marketing research, and Information categorization. 

### Dataset Description:
MUStARD(multimodal video corpus) <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a> consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which offers more details about the situation in which the utterance takes place. The dataset consists of 345 sarcastic videos and 6,020 non-sarcastic videos. The dataset is compiled from popular TV shows.

**Features of a Datapoint:**
- Video Sequence for Utterance
- Video Sequence for Context
- Transcripts of the Videos
- Speaker Name and Context character names
- Label (Sarcastic or not)

<!---
![sample_datapoint.jpg](./Images/sample_datapoint.PNG) 
|:--:| 
| **Utterance and Context Video sequences: Text and Audio-visual components (Credit: <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a>)** |

![sample_json.jpg](./Images/sample_json.PNG) 
|:--:| 
| **Example of a JSON file that contains all information about the video sequences for a datapoint (Credit: <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a>)** |
--->


<p align="center">
<img src="./Images/sample_datapoint.PNG" style="border: 1px solid black" >
<figcaption align="middle">Utterance and Context Video sequences: Text and Audio-visual components (Credit: <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a>)</figcaption>
</p>


<p align="center">
<img src="./Images/sample_json.PNG" style="border: 1px solid black" >
<figcaption align="middle">Example of a JSON file that contains all information about the video sequences for a datapoint (Credit: <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a>)</figcaption>
</p>


### Methods:
A feature vector will be generated for each datapoint that will be a combination of its audio, video, and text.
- **Video**: CNNs like VGG, ImageNet, ResNet, etc. 
- **Audio**:  RNN models and Librosa library <a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a> (MFCC, melspectogram, spectral centroid). 
- **Text**: GloVE, ELMO, Word2Vec <a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf">[1]</a>. 
<br>


The feature vectors of the 3 modalities will be combined and given as input to the supervised or unsupervised Machine Learning models. 

**Supervised methods:**
* Logistic Regression
* Naive Bayes
* Support Vector Machines
* Deep Neural Networks

**Metrics:** Precision, Recall, Accuracy, F1-Score

**Unsupervised Methods:**
* K-Mean Clustering
* Gaussian Mixture Model

**Metrics:** Silhouette Coefficient, Beta CV, pairwise measure and entropy based methods

<!---
![data_flowchart.jpg](./Images/data_flowchart.png) 
|:--:| 
| **Feature Extraction and Data Analysis Flowchart** |
--->

<p align="center">
<img src="./Images/data_flowchart.jpg" style="border: 1px solid black" >
<figcaption align="middle">Feature Extraction and Data Analysis Flowchart</figcaption>
</p>


### Potential Results and Discussion:
Our ideal goal would be to demonstrate how our multimodal approach outperforms unimodal approaches and identify the advantage of each module in providing context by experimenting with supervised and unsupervised methods. Evaluating the results of unsupervised clustering and use the observations for analyzing and cleaning the data is another task which will be addressed. Finally, will be looking to optimize and compare the different supervised classifiers using the evaluation metrics. The interpretability of the models will also be analyzed using AI Explainability tools.

### References:
<a href="https://aclanthology.org/P15-2124.pdf">[1]</a> Joshi, Aditya, Vinita Sharma, and Pushpak Bhattacharyya. "Harnessing context incongruity for sarcasm detection." Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2015. <br>
<a href="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15791781.pdf">[2]</a> Lydia Xu, Vera Xu. "Project Report: Sarcasm Detection" <br>
<a href="https://arxiv.org/pdf/1610.08815.pdf">[3]</a> Poria, Soujanya, et al. "A deeper look into sarcastic tweets using deep convolutional neural networks." arXiv preprint arXiv:1610.08815 (2016). <br>
<a href="http://www1.cs.columbia.edu/~julia/papers/teppermanetal06.pdf">[4]</a> Tepperman, Joseph, David Traum, and Shrikanth Narayanan. "" Yeah right": sarcasm recognition for spoken dialogue systems." Ninth international conference on spoken language processing. 2006.<br>
<a href="https://aclanthology.org/P14-2084">[5]</a> Byron C. Wallace, Do Kook Choe, Laura Kertz, and Eugene Charniak. 2014. "Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)". In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 512–516, Baltimore, Maryland. Association for Computational Linguistics.<br>
<a href="https://arxiv.org/pdf/1906.01815.pdf">[6]</a> Castro, Santiago, et al. "Towards multimodal sarcasm detection (an _obviously_ perfect paper)." arXiv preprint arXiv:1906.01815 (2019).

### Team Member Contributions:
<p align="center">
<img src="./Images/proposal_contribution.JPG" width="690" height="241" style="border: 1px solid black">
</p>

### Proposed Timeline:
<p align="center">
<img src="./Images/timeline.jpg" width="690" height="1000" style="border: 1px solid black">
</p>

### <a href="https://docs.google.com/spreadsheets/d/1IJ70LMrsxGJPikwkiIJkm8zs2LFqJ11leJsC7XJy1xw/edit?usp=sharing">Gantt Chart Link</a>
### <a href="https://www.canva.com/design/DAFOSFtGCEs/pXKpJgYamRmR7uMXbvFJcw/view?utm_content=DAFOSFtGCEs&utm_campaign=designshare&utm_medium=link&utm_source=viewer">Proposal Slides</a>
### <a href="https://www.youtube.com/watch?v=E7hygYxMoBk">Proposal Video - Youtube</a>


# Project MidTerm Report

## Feature Extraction
### Text
Feature extraction on Text is done using BERT. The labels are numerically encoded and then the dialogues are passed first through NLTK’s pretrained Punkt sentence tokenizer which produces a numerically encoded vector of tokens based on the tokeniser’s vocabulary, special tokens such as [CLS] and [SEP] are added, and it is suitably padded. The corresponding attention mask is generated and given as input along with the numerically encoded vector to  Pytorch pretrained BERT model that creates the vector embedding. This output is obtained from the final layer using output_all_encoded_layer=True to get the output of all the 12 layers resulting in a vector of size 768.

### Audio
Feature extraction on Audio is done using the Librosa library. First, we use the vocal separation technique implemented in [1] to extract the vocal component from the given audio file. This will ensure that any instruments or laugh tracks are removed. Then, we extract the Mel-frequency cepstral coefficients (MFCCs) and their delta, mel-scaled spectrogram and their delta and the spectral centroid of the extracted audio file. These components help in capturing audio features such as pitch, intonation, and other tonal-specific details of the speaker. We segment the audio into equal sized segments of size = 512, and we extract the above mentioned 283 features for each segment and compute the average across all segments. This serves as the feature representation of the audio file. 

<p align="center">
<img src="./Images/images/image17.png"/>
</p>


### Video
The visual features are extracted for each frame in the utterance video using a pool5 layer of a ResNet-152 model that has been pre-trained on the ImageNet dataset. Each frame of the video is first resized, and their center is cropped and the image is normalized. The processed video frame is input to the ResNet-152 model and the features are extracted. We perform the same operation for all the frames in the given video and average the features across all the frames. The resuling 2048 feature vector is the feature representation of the video file. 

## Exploratory Data Analysis (EDA)
### EDA - Text

Audio and Video files can’t be analyzed directly without pre-processing. Therefore, we perform exploratory data analysis on the audio and video files after feature extraction. We also include the extracted text features in this process. 

**Handling Missing Values:** There are no missing values in the dataset. 
**Normalizing the dataset:** We use a min-max scaler to standardize the data so that all the values are between 0 and 1. 
We perform EDA on the scaled features. 

### Correlation - Text, Audio, Video
Analyzing the correlation matrix of the text features we can observe that most features have a positive correlation with values > 0.7. For the audio features most features have a complete overlap with all values being > 0.975. For the video features some features have a correlation value < 0.4 but a good proportion of features have a correlation value > 0.4. In summary, as the features are highly correlated, they can be reduced to create a smaller set of features that can still capture the variance in the data. 

<p float="left">
<img src="./Images/images/image7.png"/>

<img src="./Images/images/image16.png"/>
</p>

<p float="left">
<img src="./Images/images/image19.png"/>

<img src="./Images/images/image24.png"/>
</p>

<p float="left">
<img src="./Images/images/image13.png"/>

<img src="./Images/images/image12.png"/>
</p>



## Feature Reduction/Selection
### Most Drifted Features
### PCA
### RF (not sure)

## Model Used
### Supervised
### Unsupervised

## Results and Discussion

## Future Directions
