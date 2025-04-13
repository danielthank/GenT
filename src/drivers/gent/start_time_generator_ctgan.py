import json
from pathlib import Path

import os
import pickle
import random
import seaborn
from copy import deepcopy
from typing import Tuple, List, Dict, Optional, Union

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot, pylab
import torch

from ctgan.data_sampler import DataSampler
from ctgan.synthesizers.ctgan import CTGAN
from drivers.gent.data import get_all_txs, get_graph_counts, ALL_TRACES
from fidelity.utils import compare_distributions
from ml.app_utils import GenTConfig
from gent_utils.utils import device

MODEL_PATH = os.path.join(os.path.dirname(__file__), "start_time_generator")
FAKE = torch.zeros((1, 1)).to(device)
REAL = torch.ones((1, 1)).to(device)
PROFILE = False

Edges = Tuple[Tuple[str, str], ...]


class StartTimesGenerator:
    def __init__(self, gen_t_config: GenTConfig, functional_loss_freq: int,
                 functional_loss_iterations: int, functional_loss_cliff: int, noise_dim: int = 32,
                 is_roll: bool = False) -> None:
        self.gen_t_config = gen_t_config
        self.noise_dim = noise_dim
        self.generator = CTGAN(
            epochs=gen_t_config.iterations, verbose=True, device=device,
            generator_dim=self.gen_t_config.generator_dim,
            discriminator_dim=self.gen_t_config.discriminator_dim,
            functional_loss=self.functional_loss, functional_loss_freq=functional_loss_freq,
            with_gcn=self.gen_t_config.with_gcn, name='start_time',
        )
        self.training_mid_data: Optional[Dict[str, Union[torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer]]] = None
        self.functional_loss_cliff = functional_loss_cliff
        self.functional_loss_iterations = functional_loss_iterations
        self.is_roll = is_roll
        self.best = None
        self.best_seed = None
        self.best_fidelity = float('inf')

        # self.load_data will load the following fields
        self.data = None
        self.n_nodes = None
        self.node_to_index = None

        self.graph_values: Optional[List[str]] = None
        self.graph_index_to_edges: Optional[Dict[int, torch.Tensor]] = None
        self.min_real_timestamp = None
        self.max_real_timestamp = None
        self.raw_timestamps_data = None

    def train(self) -> None:
        print("StartTime generator started training")
        if self.gen_t_config.start_time_with_metadata:
            return
        self.prepare_data()
        self.generator.graph_index_to_edges = self.graph_index_to_edges
        self.generator.n_nodes = self.n_nodes

        dataset = pd.DataFrame(self.data)
        dataset.columns = ["graph", "startTime"]
        graph_column = dataset["graph"].apply(self.graph_values.index)
        dataset.drop(columns="graph", inplace=True)
        dataset.insert(1, 'empty', 0)  # insert column of zeros (to allow condvec in CTGAN)

        empty_chain = pd.DataFrame(0, index=numpy.arange(len(dataset)), columns=["z"])["z"]
        if self.training_mid_data is None:
            print("StartTime generator started fitting from scratch")
            self.training_mid_data = self.generator.fit(
                dataset, graph_column, empty_chain, discrete_columns=["empty"]
            )
        else:
            print("StartTime generator started continuing fitting")
            self.generator.continue_fit(
                train_data=dataset,
                graph_data=graph_column,
                chain_data=empty_chain,
                tx_start_time=None,
                metadata=None,
                **self.training_mid_data
            )
        self.use_best()
        self.find_best_seed()

    def prepare_data(self) -> None:
        def get_graph_str(tx: dict) -> str:
            edges = {
                (
                    self.node_to_index[tx["nodesData"][n["source"]]["gent_name"]],
                    self.node_to_index[tx["nodesData"][n["target"]]["gent_name"]]
                )
                for n in tx["graph"]["edges"]
            }
            return str(tuple(sorted(list(edges))))
        all_txs = (
            get_all_txs(0, ALL_TRACES, self.gen_t_config.traces_dir)
            if self.is_roll else
            get_all_txs(self.gen_t_config.tx_start, self.gen_t_config.tx_end, self.gen_t_config.traces_dir)
        )

        # Find nodes, edges, graphs, etc. of all transactions (not only current slice)
        all_nodes = sorted(list({n["gent_name"] for tx in all_txs for n in tx["nodesData"].values()}))
        self.node_to_index = self.node_to_index or {node: index for index, node in enumerate(all_nodes)}
        self.n_nodes = len(all_nodes)
        self.graph_values: List[str] = sorted(list({get_graph_str(tx) for tx in all_txs}))
        self.graph_index_to_edges = {}
        for graph_index, graph_str in enumerate(self.graph_values):
            edges = torch.Tensor(eval(graph_str)).t().type(torch.int64).to(self.generator.get_device())
            self.graph_index_to_edges[graph_index] = edges

        data = []
        for tx in all_txs[self.gen_t_config.tx_start:self.gen_t_config.tx_end]:
            data.append((get_graph_str(tx), tx["details"]["startTime"]))

        self.data = data
        self.min_real_timestamp = min(t[1] for t in self.data)
        self.max_real_timestamp = max(t[1] for t in self.data)

    def generate_by_graph_index(self, graph_index: int, count: int):
        valid_generated_timestamps = []
        
        while len(valid_generated_timestamps) < count:
            count_to_generate = max(2, count - len(valid_generated_timestamps)) # CTGAN support generation of at least 2
            
            generated = self.generator.sample(
                graph_index_list=torch.Tensor([graph_index for _ in range(count_to_generate)]),
                chain_index_list=torch.Tensor([0 for _ in range(count_to_generate)])
            )["startTime"].values.tolist()
            
            for timestamp in generated:
                if self.min_real_timestamp <= timestamp <= self.max_real_timestamp:
                    valid_generated_timestamps.append(timestamp)
                    
                if len(valid_generated_timestamps) >= count:
                    break
        
        return valid_generated_timestamps[:count]

    def _generate_corpus(self, seed: int, use_node_indexes: bool = False) -> Dict[str, List[int]]:
        graph_counts = get_graph_counts(tx_start=self.gen_t_config.tx_start, tx_end=self.gen_t_config.tx_end, traces_dir=self.gen_t_config.traces_dir)
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

        graph_to_timestamps = {}
        for graph in sorted(graph_counts):
            count = graph_counts[graph]
            graph_edges = eval(graph)
            if isinstance(graph_edges[0][0], int) or graph_edges[0][0].isdigit():
                edge_index = graph
            else:
                edge_index = str(tuple(sorted({(self.node_to_index[source], self.node_to_index[target]) for source, target in graph_edges})))
            graph_index = self.graph_values.index(edge_index)
            graph_key = edge_index if use_node_indexes else graph
            graph_to_timestamps[graph_key] = self.generate_by_graph_index(graph_index, count=count)
        return graph_to_timestamps

    def generate_timestamps_corpus(self) -> Dict[str, List[int]]:
        if self.gen_t_config.start_time_with_metadata:
            return {}
        return self._generate_corpus(seed=self.best_seed)

    def compare(self, seed: Optional[int] = None, plot: bool = False,
                distance_per_topology: bool = False):
        if not self.data:
            self.prepare_data()

        distances = []
        all_generated = []

        seed = seed or self.best_seed
        generated_corpus = self._generate_corpus(seed, use_node_indexes=True)
        for graph_index in range(len(self.graph_values)):
            edges = self.graph_values[graph_index]
            real = [t[1] for t in self.data if t[0] == edges]
            if not real:
                # This graph does not appear in the current slice of data
                continue
            generated = generated_corpus[edges]

            min_value = min(self.min_real_timestamp, *generated)
            max_value = max(self.max_real_timestamp, *generated)
            real_normalized = [(v - min_value) / (max_value - min_value) for v in real]
            generated_normalized = [(v - min_value) / (max_value - min_value) for v in generated]
            if distance_per_topology:
                distances.append(compare_distributions(real_normalized, generated_normalized))
            all_generated.extend(generated)
            if plot:
                fig, ax = pyplot.subplots()
                bins = np.histogram(np.hstack((real, generated)), bins=10)[1]
                ax.hist(real, bins=bins, alpha=0.5, label="real")
                ax.hist(generated, bins=bins, alpha=0.5, label="generated")
                ax.set_xlim(min_value, max_value)
                ax.legend()
        if plot:
            pyplot.show()
        if not distance_per_topology:
            reals = [t[1] for t in self.data]
            all_generated = [(v - self.min_real_timestamp) / (self.max_real_timestamp - self.min_real_timestamp) for v in all_generated]
            reals = [(v - self.min_real_timestamp) / (self.max_real_timestamp - self.min_real_timestamp) for v in reals]
            return [compare_distributions(all_generated, reals)], seed, all_generated
        return distances, seed

    def get_graphite_score(self, data_size: int) -> float:
        """
        Execute the graphite train.py with our changes before running this function.
        """
        results = []
        graphite = eval(open('/tmp/graphite_ts.txt', 'r').read())
        real_normalized = [(v[1] - self.min_real_timestamp) / (self.max_real_timestamp - self.min_real_timestamp) for v in self.data]
        results.append(compare_distributions(graphite, real_normalized))
        graphite *= int(data_size / len(graphite))  # Graphite generates less samples
        return compare_distributions(graphite, real_normalized), graphite


    def get_random_score(self, iterations: int = 10) -> float:
        reals = [t[1] for t in self.data]
        results = []
        for _ in range(iterations):
            rnd_timestamps = list(numpy.random.randint(self.min_real_timestamp, self.max_real_timestamp, size=(len(reals))))
            results.append(compare_distributions(rnd_timestamps, reals))
        return numpy.average(results)

    def functional_loss(self, iteration_index: int):
        if iteration_index < self.functional_loss_cliff:
            return
        for repeat_index in range(self.functional_loss_iterations):
            seed = int(random.random() * (2 ** 32))
            distances, seed, _ = self.compare(seed=seed)
            avg = numpy.average(distances)
            if avg < self.best_fidelity:
                self.best = deepcopy(self.generator._generator.state_dict())
                self.best_fidelity = avg
                self.best_seed = seed
                print(f"Best loss: ({iteration_index}/{repeat_index}): {avg} ({seed})", end=", ")

    def use_best(self):
        self.generator._generator.load_state_dict(self.best)
        # TODO: do we need to use the discriminator as well?

    def save(self, path=MODEL_PATH):
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.generator, open(f"{path}/generator_all.pkl", "wb"))
        # This is a hack to make the model smaller
        sampler, self.generator._data_sampler = self.generator._data_sampler, DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
        self.generator._noise.graph_data, self.generator._noise.chain_data, self.generator._noise.trigger_data, self.generator._noise.tx_start_time = None, None, None, None
        self.generator.__doc__ = None
        self.generator.functional_loss = None
        self.generator.save(f"{path}/start_time_ctgan_generator.pkl")
        self.generator._data_sampler = sampler
        pickle.dump(self.generator.graph_index_to_edges, open(f"{path}/graph_index_to_edges.pkl", "wb"))
        pickle.dump(self.node_to_index, open(f"{path}/node_to_index.pkl", "wb"))
        pickle.dump(self.best_seed, open(f"{path}/best_seed.pkl", "wb"))
        pickle.dump(self.graph_values, open(f"{path}/graph_values.pkl", "wb"))
        pickle.dump(self.min_real_timestamp, open(f"{path}/min_real_timestamp.pkl", "wb"))
        pickle.dump(self.max_real_timestamp, open(f"{path}/max_real_timestamp.pkl", "wb"))

    def save_local(self, path=MODEL_PATH):
        pickle.dump(self.training_mid_data, open(f"{path}/local.pkl", "wb"))

    def load(self, path=MODEL_PATH):
        if self.gen_t_config.start_time_with_metadata:
            return
        self.generator = CTGAN.load(f"{path}/start_time_ctgan_generator.pkl")
        self.generator._device = device
        self.generator._noise.device = device
        self.generator._data_sampler = DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
        if os.path.exists(f"{path}/graph_index_to_edges.pkl"):
            self.generator.graph_index_to_edges = pickle.load(open(f"{path}/graph_index_to_edges.pkl", "rb"))
            self.node_to_index = pickle.load(open(f"{path}/node_to_index.pkl", "rb"))
        else:
            metadata_path = os.path.join(path, '..', 'metadata')
            self.generator.graph_index_to_edges = pickle.load(open(f"{metadata_path}/graph_index_to_edges.pkl", "rb"))
            self.node_to_index = pickle.load(open(f"{metadata_path}/node_to_index.pkl", "rb"))
        self.best_seed = pickle.load(open(f"{path}/best_seed.pkl", "rb"))
        self.graph_values = pickle.load(open(f"{path}/graph_values.pkl", "rb"))
        self.min_real_timestamp = pickle.load(open(f"{path}/min_real_timestamp.pkl", "rb"))
        self.max_real_timestamp = pickle.load(open(f"{path}/max_real_timestamp.pkl", "rb"))

    def load_all(self, path=MODEL_PATH) -> "StartTimesGenerator":
        self.load(path)
        self.is_roll = True
        self.generator = pickle.load(open(f"{path}/generator_all.pkl", "rb"))
        self.generator.functional_loss = self.functional_loss
        self.training_mid_data = pickle.load(open(f"{path}/local.pkl", 'rb'))
        return self

    def find_best_seed(self):
        print("Prev best fidelity:", self.best_fidelity, self.best_seed, end=", ")
        self.best_fidelity = self.compare(seed=self.best_seed)[0][0]
        print("First best:", self.best_fidelity, self.best_seed, end=", ")
        for _ in range(min(30, self.gen_t_config.iterations)):
            print('.', end='', flush=True)
            seed = int(random.random() * (2 ** 32))
            dist = self.compare(seed=seed)[0][0]
            if dist < self.best_fidelity:
                self.best_fidelity = dist
                self.best_seed = seed
                print("New best:", self.best_fidelity, seed, end=", ")

    @staticmethod
    def get(gen_t_config: GenTConfig, is_roll: bool = False) -> "StartTimesGenerator":
        return StartTimesGenerator(
            gen_t_config,
            functional_loss_freq=1,
            functional_loss_iterations=1,
            functional_loss_cliff=min(100, gen_t_config.iterations // 2),
            is_roll=is_roll,
        )


def train_and_save(gen_t_config: GenTConfig, path: Union[str, Path], is_roll: bool = False):
    """
    This function is here to support multiprocessing
    """
    if gen_t_config.start_time_with_metadata:
        return
    gen = StartTimesGenerator.get(gen_t_config, is_roll=is_roll)
    assert gen_t_config.iterations >= gen.functional_loss_iterations
    gen.train()
    gen.save(path=str(path))
    gen.save_local(path=str(path))
    print("Done train_and_save start_time fidelity:", gen.best_fidelity)


def continue_train_and_save(gen_t_config: GenTConfig, path: Union[str, Path], from_path: Union[str, Path]):
    """
    This function is here to support multiprocessing
    """
    gen = StartTimesGenerator.get(gen_t_config, is_roll=True)
    gen.load_all(path=str(from_path))
    gen.train()
    gen.save(path=str(path))
    gen.save_local(path=str(path))
    print("\nStart time root fidelity:", gen.best_fidelity, gen.best_seed)


def plot_graphit_vs_gent():
    seaborn.set()
    seaborn.set_style("whitegrid")
    pylab.rcParams.update(
        {
            "figure.figsize": (7, 4),
            "figure.titlesize": 14,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "lines.linewidth": 2,
            "legend.fontsize": 20,
        }
    )
    if not os.path.exists('graphite_test_cache.txt'):
        generator = StartTimesGenerator(GenTConfig(iterations=10, tx_end=ALL_TRACES),
                                        functional_loss_freq=1, functional_loss_iterations=3,
                                        functional_loss_cliff=1)
        generator.train()
        generator.find_best_seed()
        syn_score, _, synthetic = generator.compare(seed=generator.best_seed, plot=False)

        real = [(v[1] - generator.min_real_timestamp) / (generator.max_real_timestamp - generator.min_real_timestamp) for v in generator.data]
        graphite_score, graphite = generator.get_graphite_score(data_size=len(real))
        print("graphite score", graphite_score)
        print("synthetic score", syn_score)

        with open('graphite_test_cache.txt', 'w') as f:
            f.write(json.dumps({"graphite": graphite, "synthetic": synthetic, "real": real,}))
    else:
        with open('graphite_test_cache.txt', 'r') as f:
            data = json.loads(f.read())
            graphite = data["graphite"]
            synthetic = data["synthetic"]
            real = data["real"]

    fig, ax = pyplot.subplots()
    bins = np.histogram(np.hstack((real, synthetic, graphite)), bins=10)[1]
    ax.hist(real, bins=bins, alpha=0.7, label="real")
    ax.hist(synthetic, bins=bins, alpha=0.7, label="Gen-T")
    ax.hist(graphite, bins=bins, alpha=0.7, label="Graphite")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized timestamp")
    ax.set_ylabel("Number of traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(r"../../paper/figures/graphite_vs_gent.pdf", format="pdf", bbox_inches='tight')
    pyplot.show()


def main(train: bool = True):
    generator = StartTimesGenerator(GenTConfig(iterations=10, tx_end=ALL_TRACES),
                                    functional_loss_freq=1, functional_loss_iterations=3,
                                    functional_loss_cliff=1)
    if train:
        generator.train()
        generator.save()
    generator.load()

    generator.find_best_seed()

    dist = generator.compare(seed=generator.best_seed, plot=True)[0][0]
    print("Best seed:", generator.best_seed, "Distance:", dist)


if __name__ == '__main__':
    if PROFILE:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

    # main()
    # build_sample_data()
    # roll_test()
    plot_graphit_vs_gent()

    if PROFILE:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
