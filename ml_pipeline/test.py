from config import Config

class Test:
    def __init__(self, config: Config) -> None:
        self.config = config

    def test(self, model, test_input):
        test_pred=model.predict(test_input)

        return test_pred
