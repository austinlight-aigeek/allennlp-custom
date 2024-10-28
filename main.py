from srl_bert import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)
sentence = "The quick brown fox jumps over the lazy dog."

example = predictor.predict(sentence)

print(example)
