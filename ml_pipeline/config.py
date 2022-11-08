class Config:
    MODEL="SVM"
    USE_BERT=True
    EMBEDDING_DIM = 300  # GloVe embedding size
    WORD_EMBEDDING_PATH = "embeddings/glove.840B.300d.txt"
    MAX_SENT_LENGTH = 20
    NUM_CLASSES = 2  # Binary classification of sarcasm
    VALIDATION_SET_SIZE = 0.1
    TEST_SET_SIZE = 0.1
    SVM_C = 10.0
    FRAMES_DIR_PATH = "data/frames/utterances_final"
    