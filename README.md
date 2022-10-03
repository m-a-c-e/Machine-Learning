# ML 7641: Team 21
## Project Title: MultiModel Sarcasm Detection

### Introduction:
Sarcasm detection is a task under sentiment analysis that involves recognizing sarcastic utterances whose perlocutionary effect requires a great understanding of the conversational context, the utterance, and some basic knowledge to perceive the whole conversation. However, the ironic and metaphorical nature of sarcasm poses a challenge for computer models that try to analyze the sentiment of the conversation.

### Problem Definition:
There have been several NLP models that have tried to detect sarcasm in text statements either using annotation or incongruity. Though these models have managed to achieve good accuracy by including context and embeddings of personality traits[1], the existing models do not focus on sarcasm detection in conversation. Past work in sarcasm detection using speech is based on intonation that corresponds to sarcastic behavior[2]. Visual markers to understand sarcasm have been studied rarely. Thus, through our model, we aim to detect sarcasm in video segments of dialogues by extracting visual features, and aligning it with corresponding textual and audio cues to derive additional contextual information to identify sarcastic behavior. The motivation for sarcasm detection is to improve Conversational AI’s ability to perceive human interaction, Opinion mining, Marketing research, and Information categorization [3]. 

### Dataset Description:
MUStARD(multimodal video corpus) consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which offers more details about the situation in which the utterance takes place. The dataset consists of 345 sarcastic videos and 6,020 non-sarcastic videos. The dataset is compiled from popular TV shows like Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anonymous. 

Sample DataPoint: 
- Video Sequence for Utterance
- Video Sequence for Context
- Transcripts of the Videos
- Speaker Name and Context character names
- Label (Sarcastic or not)

![My Image](Images/sample_json.png)

