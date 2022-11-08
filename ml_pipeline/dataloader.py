from config import Config
from sklearn.model_selection import train_test_split

#read the raw video 
#read the raw audio features
#read the raw text features
#preprocess video and get video feature vector
#preprocess audio and get audio feature vector
#preprocess text and get text feature vector
#pca
#concatenate audio, video and text
#split into train, test and validation sets

class Dataloader:

    def __init__(self, video_file_path, audio_file_path, text_file_path):
        self.video_path = video_file_path
        self.audio_path = audio_file_path
        self.text_path = text_file_path

    def extract_video_features(self):
        pass

    def extract_audio_features(self):
        pass

    def extract_text_features(self):
        pass

    def concatenate_features(self, video_features, audio_features, text_features, labels):
        self.X = []
        self.y = labels
        pass

    def split_dataset(self):
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, \
            test_size=Config.TEST_SET_SIZE, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, \
            test_size=Config.VALIDATION_SET_SIZE, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
