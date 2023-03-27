import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)

        # Compute precision and recall scores
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        return loss, {"accuracy": accuracy}
        # return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=5,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
# if __name__ == "_main_":
#     model = LogisticRegression()
#     utils.set_initial_params(model)
#     lambd = 0.1  # regularization strength
#     mu = 1.0  # proximal term strength

#     def on_fit_config_fn(num_epochs):
#         return {
#             "c": lambd,
#             "mu": mu,
#             "batch_size": 32,
#             "epochs": num_epochs,
#         }

#     strategy = fl.server.strategy.FedProx(
#         on_fit_config_fn=on_fit_config_fn,
#         min_available_clients=5,
#         evaluate_fn=get_evaluate_fn(model),
#         initial_parameters=utils.get_model_parameters(model),
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         strategy=strategy,
#         config=fl.server.ServerConfig(num_rounds=5),
#     )