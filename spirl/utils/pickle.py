import pickle


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
