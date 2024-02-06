import copy
import math
from typing import List, Tuple, Dict

import flwr as fl
import numpy as np
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from sklearn.neighbors import KNeighborsClassifier

from ECG import DataPreprocessor
from flower_client import FlowerClient
from parameter import *


def get_client_fn(dataset_partitions, knn_model, defense_strategy):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> FlowerClient:
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
        return FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid, cid, knn_model, defense_strategy)

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
            config: Dict[str, fl.common.Scalar]):
        model = get_model(x_test)
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate


def partition(x_train, y_train):
    """Partitions the dataset."""
    partitions = []
    # We keep all partitions equal-sized in this example
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    for cid in range(NUM_CLIENTS):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
        partitions.append((x_train[idx_from:idx_to] / 255.0, y_train[idx_from:idx_to]))
    return partitions


def create_global_knn(x_train, y_train) -> KNeighborsClassifier:
    """Create and train a global KNN model."""
    knn = KNeighborsClassifier(3, weights="distance")
    knn.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    return knn


# *defense strategies*:
# 0: no defense
# 1: substitution
# 2: deletion
# 3: data augmentation
##
def get_input():
    """Prompt user for custom parameters."""
    while True:
        attacked_clients = []
        defense_strategy = 0
        attacked_percentage = 0.0
        attacked_input = input("Enter 'True' if attacked, 'False' otherwise: ")
        if attacked_input in ['True', 'False']:
            attacked = (attacked_input == 'True')
            valid_input_clients = False
            while not valid_input_clients:
                try:
                    clients_attacked = int(input("Enter the number of attacked clients (1-3): "))
                    if 1 <= clients_attacked <= 3:
                        if clients_attacked == 1:
                            attacked_clients = [0]
                        elif clients_attacked == 2:
                            attacked_clients = [0, 1]
                        elif clients_attacked == 3:
                            attacked_clients = [0, 1, 2]
                        break
                    else:
                        print("Clients attacked must be between 1 and 3.")
                except ValueError:
                    print("Please enter a valid integer.")
            valid_input_defense = False
            while not valid_input_defense:
                try:
                    defense_strategy = int(input("Enter the defense strategy (0 for no defense, 1 for substitution, "
                                                 "2 for deletion, 3 for data augmentation): "))
                    if defense_strategy in [0, 1, 2, 3]:
                        break
                    else:
                        print("Defense strategy must be in [0, 1, 2, 3].")
                except ValueError:
                    print("Please enter a valid integer.")
            valid_input_percentage = False
            while not valid_input_percentage:
                try:
                    attacked_percentage = float(input("Enter the attack percentage: "))
                    if 0 <= attacked_percentage <= 1:
                        break
                    else:
                        print("Attack percentage must be between 0 and 1.")
                except ValueError:
                    print("Please enter a valid floating point number.")
            break
        else:
            print("Please enter 'True' or 'False'.")

    return attacked, defense_strategy, attacked_clients, attacked_percentage


def main() -> None:
    """Main function to start the simulation."""
    running = True
    print("Starting simulation, please enter your parameters")
    while running:
        attacked, defense_strategy, clients_attacked, attacked_percentage = get_input()
        run(8, attacked, defense_strategy, clients_attacked, attacked_percentage)

        valid_input = False
        while not valid_input:
            try:
                still_running = int(input("Do you want to continue? (0 for yes, 1 for no)"))
                if still_running in [0, 1]:
                    running = still_running == 0
                    valid_input = True
                else:
                    print("Your input must be 0 or 1.")
            except ValueError:
                print("Please enter a valid integer.")
    print("Thank you")


def run(num_rounds, attacked, defense_strategy, clients_attacked, attack_percentage) -> None:
    """Run the simulation."""
    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    partitions = [DataPreprocessor([232, 222, 209, 201, 207, 118, 220, 223, 202, 234, 124]).split_data(),
                  DataPreprocessor([100, 200, 213, 210, 114, 219, 233, 113, 108, 101, 205]).split_data(),
                  DataPreprocessor([215, 228, 103, 112, 203, 208, 116, 117, 121, 231, 102]).split_data()]

    features_test = DataPreprocessor([104, 105, 106, 217, 107, 109, 115, 221, 119, 123, 214]).get_test_data()

    # Creo un KNN sui dati puliti
    # Sviluppi futuri migliorare knn con i round similmente al modello centrale
    x_train = []
    y_train = []
    for tuple_pair in partitions:
        x_train.extend(tuple_pair[0])
        y_train.extend(tuple_pair[1])

    global_knn_model = create_global_knn(np.array(x_train), np.array(y_train))

    NUM_ROUNDS = num_rounds

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.1,
    }

    if attacked:
        partitions = attack_clients(partitions, clients_attacked, attack_percentage)

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
        client_fn=get_client_fn(partitions, global_knn_model, defense_strategy),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
        }
    )


def attack_clients(partitions, attacked_clients, attack_percentage):
    for cid in attacked_clients:
        # Flip labels for a certain percentage of data in each attacked client
        flipped, modified = label_flipping_attack(partitions[cid], attack_percentage)
        partitions[cid] = flipped

    return partitions


def label_flipping_attack(part, attack_percentage, random_seed=42):
    """
    Apply label flipping attack to a partition.
    """

    np.random.seed(random_seed)
    labels = copy.deepcopy(part[1])

    # Determine the number of samples to be attacked based on the attack_percentage
    num_samples = len(labels)
    num_attacked_samples = int(attack_percentage * num_samples)

    # Generate random indices to flip the labels
    attacked_indices = np.random.choice(num_samples, num_attacked_samples, replace=False)

    # Generate new random labels to flip to (assuming labels are integers)
    new_labels = np.random.randint(2, 4, num_attacked_samples)

    # Flip labels for selected samples
    labels[attacked_indices] = new_labels

    success_flag = not np.all(labels == part[1])

    # Update the partition with the modified labels
    modified_partition = (part[0], labels)

    return modified_partition, success_flag


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()
