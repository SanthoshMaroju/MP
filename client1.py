import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
from imblearn.over_sampling import ADASYN,SMOTE
smote = SMOTE(sampling_strategy=0.7)
adasyn= ADASYN(sampling_strategy=0.7)
from collections import Counter
import utils

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = utils.load_data()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    (X_train, y_train) = utils.partition(X_train, y_train, 5)[0]

    print("Client1: ", Counter(y_train),sep=" ")

    # X_train_smote, Y_train_smote = smote.fit_resample(X_train, y_train)

    X_train_smote, Y_train_smote = adasyn.fit_resample(X_train, y_train)

    X_train=np.concatenate((X_train, X_train_smote), axis=0)
    y_train=np.concatenate((y_train, Y_train_smote), axis=0) 

    print("After Client1: ", Counter(y_train),sep=" ")

    model = LogisticRegression( )

    utils.set_initial_params(model)

    # Define Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)

            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
