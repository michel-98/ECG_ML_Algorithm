import copy
import math
from typing import List, Tuple, Dict

import flwr as fl
import numpy as np
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from ECG import DataPreprocessor
from flower_client import FlowerClient
from parameter import *


def get_client_fn(dataset_partitions, knn_model):
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
        return FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid, cid, knn_model)

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
    """Return an evaluation function for server-side (i.e., centralized) evaluation."""
    x_test, y_test = testset

    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ):
        model = get_model(x_test)
        model.set_weights(parameters)
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


def create_global_Knn(x_train, y_train, testset) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(3, weights="distance")
    x_test, y_test = testset
    knn.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    # y_pred_test = knn.predict(x_test.reshape(x_test.shape[0], -1))
    # print("Classification Report on Training Data:")
    # print(classification_report(y_test, y_pred_test))
    # plot_labels_comparison(y_test, y_pred_test)
    return knn


def plot_labels_comparison(y_train, y_train_predicted):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original Labels (y_train)')
    plt.hist(y_train, bins=2, color='blue', alpha=0.7)
    plt.xlabel('Labels')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.title('Corrected Labels (y_train_predicted)')
    plt.hist(y_train_predicted, bins=2, color='green', alpha=0.7)
    plt.xlabel('Labels')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def main() -> None:
    run(8)

    # Defense works well on more than 2 client poisoned


def run(num_rounds) -> None:
    # Parse input arguments
    args = parser.parse_args()

    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    partitions = [DataPreprocessor([232, 222, 209, 201, 207, 118, 220, 223, 202, 234, 124]).split_data(),
                  DataPreprocessor([100, 200, 213, 210, 114, 219, 233, 113, 108, 101, 205]).split_data(),
                  DataPreprocessor([215, 228, 103, 112, 203, 208, 116, 117, 121, 231, 102]).split_data()]

    features_test = DataPreprocessor([104, 105, 106, 217, 107, 109, 115, 221, 119, 123, 214]).get_test_data()

    ## Creo un KNN sui dati puliti
    ## Sviluppi futuri migliorare knn con i round similmente al modello centrale
    x_train = []
    y_train = []
    for tuple_pair in partitions:
        x_train.extend(tuple_pair[0])
        y_train.extend(tuple_pair[1])

    global_knn_model = create_global_Knn(np.array(x_train), np.array(y_train), features_test)

    NUM_ROUNDS = num_rounds

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.1,
    }

    if True:
        print("attaccato")
        partitions = attackClients(partitions)

    # doSomething(partitions, global_knn_model)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
        min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(features_test),  # global evaluation function
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(partitions, global_knn_model),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
        }
    )


def doSomething(partitions, knn: KNeighborsClassifier):
    # Extract partition for client with id = cid
    x_train, y_train = partitions[int(0)]
    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(x_train) * 0.9)
    x_train_cid, y_train_cid = (
        x_train[:split_idx],
        y_train[:split_idx],
    )
    x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]

    # Create and return client
    flwClt = FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid, 0, knn)
    flwClt.fit(flwClt.get_parameters(None), None)


def attackClients(partitions):
    attacked_clients = [1]  # Index of clients to be attacked
    for cid in attacked_clients:
        # Flip labels for a certain percentage of data in each attacked client
        flipped, modified = label_flipping_attack(partitions[cid], attack_percentage=0.8)
        partitions[cid] = flipped
        print("Client attacked: " + str(cid) + " partition modified: " + str(modified))
    return partitions


def label_flipping_attack(partition, attack_percentage=0.5, random_seed=42):
    """
    Apply label flipping attack to a partition.

    Parameters:
    - partition: Tuple containing (data, labels)
    - attack_percentage: Percentage of labels to be flipped

    Returns:
    - partition with labels flipped
    """
    # Extract labels from the partition
    np.random.seed(random_seed)
    labels = copy.deepcopy(partition[1])

    # Determine the number of samples to be attacked based on the attack_percentage
    num_samples = len(labels)
    num_attacked_samples = int(attack_percentage * num_samples)

    # Generate random indices to flip the labels
    attacked_indices = np.random.choice(num_samples, num_attacked_samples, replace=False)

    # Generate new random labels to flip to (assuming labels are integers)
    new_labels = np.random.randint(2, 4, num_attacked_samples)

    # Flip labels for selected samples
    labels[attacked_indices] = new_labels

    success_flag = not np.all(labels == partition[1])

    # Update the partition with the modified labels
    modified_partition = (partition[0], labels)

    return modified_partition, success_flag


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()
