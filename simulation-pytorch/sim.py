import math
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from ECG import DataPreprocessor
from flower_client import FlowerClient
from parameter import *


def get_client_fn(dataset_partitions):
    """Return a function to construc a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        x_train, y_train = dataset_partitions[int(cid)]
        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(x_train) * 0.9)
        x_train_cid, y_train_cid = (
            x_train[:split_idx],
            y_train[:split_idx],
        )
        x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]

        # Create and return client
        return FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid)

    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics.

    It will aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    x_test, y_test = testset

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ):
        model = get_model(x_test)  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate


def partition(x_train, y_train):
    """Download and partitions the MNIST dataset."""
    partitions = []
    # We keep all partitions equal-sized in this example
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    for cid in range(NUM_CLIENTS):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
        partitions.append((x_train[idx_from:idx_to] / 255.0, y_train[idx_from:idx_to]))
    return partitions


def main() -> None:
    run(10)


def run(num_rounds) -> None:
    # Parse input arguments
    args = parser.parse_args()

    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    partitions = [DataPreprocessor([232, 222, 209, 201, 207, 118, 220, 223, 202, 234, 124]).split_data(),
                  DataPreprocessor([100, 200, 213, 210, 114, 219, 233, 113, 108, 101, 205]).split_data(),
                  DataPreprocessor([215, 228, 103, 112, 203, 208, 116, 117, 121, 231, 102]).split_data()]

    features_test = DataPreprocessor([104, 105, 106, 217, 107, 109, 115, 221, 119, 123, 214]).get_test_data()

    NUM_ROUNDS = num_rounds

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=3,  # Never sample less than 10 clients for training
        min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(features_test),  # global evaluation function
    )

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.1,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(partitions),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()
