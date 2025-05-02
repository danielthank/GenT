import os
import pickle
import random
import numpy
import numpy as np
import pandas as pd
import torch

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from copy import deepcopy
from typing import Tuple, List, Dict, Optional, Union
from matplotlib import pyplot
from pathlib import Path
from drivers.gent.data import get_all_txs, ALL_TRACES
from ml.app_utils import GenTConfig
from gent_utils.utils import device

Edges = Tuple[Tuple[str, str], ...]


class StartTimesGenerator:
    def __init__(self, gen_t_config: GenTConfig, functional_loss_freq: int,
                 functional_loss_iterations: int, functional_loss_cliff: int, noise_dim: int = 32,
                 is_roll: bool = False) -> None:
        self.gen_t_config = gen_t_config
        self.models_dir = os.path.join(gen_t_config.models_dir, "start_time")
        self.noise_dim = noise_dim
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
        self.raw_timestamps_data = None

    def train(self) -> None:
        print("StartTime generator started training")
        if self.gen_t_config.start_time_with_metadata:
            return
        self.prepare_data()
        dataset = pd.DataFrame(self.data)
        dataset.columns = ["graph", "startTime"]
        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("startTime", sdtype="numerical")
        self.generator = CTGANSynthesizer(
            epochs=self.gen_t_config.iterations,
            batch_size=self.gen_t_config.batch_size,
            metadata=metadata,
            enforce_min_max_values=True,
            verbose=True
        )
        self.generator.fit(dataset)

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
        self.graph_counts = {graph: 0 for graph in self.graph_values}
        for tx in all_txs:
            graph_str = get_graph_str(tx)
            self.graph_counts[graph_str] += 1
        self.graph_index_to_edges = {}
        for graph_index, graph_str in enumerate(self.graph_values):
            edges = torch.Tensor(eval(graph_str)).t().type(torch.int64).to('cuda')
            self.graph_index_to_edges[graph_index] = edges

        data = []
        for tx in all_txs[self.gen_t_config.tx_start:self.gen_t_config.tx_end]:
            data.append((get_graph_str(tx), tx["details"]["startTime"]))

        self.data = data

    def _generate_corpus(self) -> Dict[str, List[int]]:
        rows = []
        for graph, count in self.graph_counts.items():
            rows.extend([{"graph": graph} for _ in range(count)])
        known_columns = pd.DataFrame(rows)
        return self.generator.sample_remaining_columns(
            known_columns=known_columns,
            max_tries_per_batch=500
        )

    def generate_timestamps_corpus(self) -> Dict[str, List[int]]:
        if self.gen_t_config.start_time_with_metadata:
            return {}
        return self._generate_corpus()

    def compare(self, plot: bool = False):
        if not self.data:
            self.prepare_data()
        
        generated_corpus = self._generate_corpus().to_records(index=False)
        for graph in self.graph_values:
            generated = [t[1] for t in generated_corpus if t["graph"] == graph]
            real = [t[1] for t in self.data if t[0] == graph]
            if plot:
                fig, ax = pyplot.subplots()
                bins = np.histogram(np.hstack((real, generated)), bins=10)[1]
                ax.hist(real, bins=bins, alpha=0.5, label="real")
                ax.hist(generated, bins=bins, alpha=0.5, label="generated")
                ax.legend()
        pyplot.show()

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

    def save(self):
        path = self.models_dir
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.generator, open(f"{path}/generator_all.pkl", "wb"))
        # This is a hack to make the model smaller
        # sampler, self.generator._data_sampler = self.generator._data_sampler, DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
        self.generator.__doc__ = None
        self.generator.save(f"{path}/start_time_ctgan_generator.pkl")
        # self.generator._data_sampler = sampler
        pickle.dump(self.graph_index_to_edges, open(f"{path}/graph_index_to_edges.pkl", "wb"))
        pickle.dump(self.node_to_index, open(f"{path}/node_to_index.pkl", "wb"))
        pickle.dump(self.graph_values, open(f"{path}/graph_values.pkl", "wb"))

    def load(self):
        path = self.models_dir
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
    gen.save()
    print("Done train_and_save start_time fidelity:", gen.best_fidelity)
