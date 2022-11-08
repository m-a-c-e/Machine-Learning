from config import Config
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
class Train:

    def __init__(self, config: Config) -> None:
        self.config = config
        
    def train_svm(self, train, train_labels):
        clf = svm.SVC(kernel='rbf', gamma="scale", C=Config.SVM_C)
        clf.fit(train, train_labels)
        return clf

    def train_random_forest(self, train, train_labels):
        
        pass


    def train(self, train_set, train_labels, model="SVM"):
        
        #split into train and test sets
        if model == "SVM":
            svm_model = self.train_svm(train_set, train_labels)
            return svm_model

        elif model=="RandomForest":
            random_forest_model = self.train_random_forest(train_set, train_labels)
            return random_forest_model
            



    
        
