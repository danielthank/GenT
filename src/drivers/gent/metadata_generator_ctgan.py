from pathlib import Path

import json
import pickle
import random
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, List, Tuple, Union, Set
import os

import numpy
import numpy as np
import pandas as pd
import torch
from torch.multiprocessing import set_start_method

from ctgan import CTGAN
from ctgan.data_sampler import DataSampler

from drivers.gent.data import get_full_dataset_chains, edge_index_to_graph, get_graph_counts, ALL_TRACES
from fidelity.utils import compare_distributions
from ml.app_denormalizer import prepare_components, prepare_tx_structure
from ml.app_utils import GenTConfig, get_key_name, get_key_value
from gent_utils.utils import NpEncoder, device

PROFILE = False
warnings.filterwarnings('ignore', module='rdt')
try:
    set_start_method('spawn')  # To handle multiprocessing with pytorch
except RuntimeError:
    pass

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def get_node_columns(index: int, with_start_time: bool = False):
    metadata_columns = [
        'gapFromParent_{i}', 'duration_{i}', 'hasError_{i}', 'metadata_{i}_0', 'metadata_{i}_1',
        'metadata_{i}_2', 'metadata_{i}_3', 'metadata_{i}_4'
    ]
    if with_start_time:
        metadata_columns.append('txStartTime')
    return [c.format(i=index) for c in metadata_columns]


def to_metadata_normalized_columns(df: pd.DataFrame, index: int) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df).transpose()
    metadata_columns = [c.replace(f'_{index}', '', 1) for c in df.columns]
    df.columns = metadata_columns
    return df


class MetadataGenerator:
    def __init__(self, gen_t_config: GenTConfig, functional_loss_freq: int,
                 functional_loss_iterations: int, functional_loss_cliff: int, is_roll: bool = False) -> None:
        self.gen_t_config = gen_t_config
        self.n_epochs = self.gen_t_config.iterations
        self.root_generator: Optional[CTGAN] = None
        self.chained_generator: Optional[CTGAN] = None
        self.graph_index_to_chains: Dict[int, Tuple[List[int], List[int]]] = {}
        self.column_to_values = {}
        self.node_to_index = {}
        self.comparison_data: Optional[Dict[str, pd.DataFrame]] = None
        self.functional_loss_freq = functional_loss_freq
        self.functional_loss_cliff = functional_loss_cliff
        self.functional_loss_iterations = functional_loss_iterations
        self.is_roll = is_roll
        self.best = None
        self.best_noise = None
        self.best_root_seed = None
        self.best_chained_seed = None
        self.best_fidelity = float('inf')
        assert self.gen_t_config.chain_length >= 2, "second node is conditioned by the first"
        self.root_training_mid_data: Optional[
            Dict[str, Union[torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer]]] = None
        self.chain_training_mid_data: Optional[
            Dict[str, Union[torch.nn.Module, torch.optim.Optimizer, torch.optim.Optimizer]]] = None

    def train(self):
        self.train_root()
        self.train_chained()

    def prepare(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str], int, Dict[int, torch.Tensor]]:
        dataset, all_dataset = get_full_dataset_chains(self.gen_t_config, load_all=self.is_roll)
        dataset = dataset.copy()

        self.column_to_values = {
            "graph": list(all_dataset["graph"].unique()),
            "chain": list(all_dataset["chain"].unique()),
        }
        all_nodes = sorted({a for g in self.column_to_values["graph"] for a, b in eval(g)} \
            .union({b for g in self.column_to_values["graph"] for a, b in eval(g)}))
        self.node_to_index = {node_name: i for i, node_name in enumerate(all_nodes)}

        self.graph_index_to_chains = {index: (
            all_dataset[(all_dataset["graph"] == graph) & (all_dataset["is_root_chain"] == True)]["chain"].apply(self.column_to_values["chain"].index).unique().tolist(),
            all_dataset[(all_dataset["graph"] == graph) & (all_dataset["is_root_chain"] == False)]["chain"].apply(self.column_to_values["chain"].index).unique().tolist(),
        ) for index, graph in enumerate(self.column_to_values["graph"])}

        self.best_fidelity = float('inf')
        self.best = None
        self.best_noise = None

        dataset["graph"] = dataset["graph"].apply(self.column_to_values["graph"].index)
        dataset["chain"] = dataset["chain"].apply(self.column_to_values["chain"].index)
        str_columns = []
        for column in dataset.columns:
            if not isinstance(dataset[column][0], (int, float, np.int64, np.bool_)):
                str_columns.append(column)

        graph_index_to_edges = {}
        for graph in all_dataset["graph"]:
            edges = eval(graph)
            edges = [(self.node_to_index[source], self.node_to_index[target]) for source, target in edges]
            edges = torch.Tensor(edges).t().type(torch.int64).to(device)
            graph_index_to_edges[self.column_to_values["graph"].index(graph)] = edges

        tx_start_time = dataset["txStartTime"]
        if not self.gen_t_config.start_time_with_metadata:
            dataset.drop(columns="txStartTime", inplace=True)
        graph_column = dataset["graph"]
        dataset.drop(columns="graph", inplace=True)
        chain_column = dataset["chain"]
        dataset.drop(columns="chain", inplace=True)

        return dataset, graph_column, chain_column, tx_start_time, str_columns, len(all_nodes), graph_index_to_edges

    def prepare_comparison_data(self, dataset, chain_column, is_root: bool):
        chain_index_to_node = {chain_index: self.column_to_values["chain"][chain_index].split('#') for chain_index in chain_column.unique()}
        chain_index_to_column = defaultdict(dict)
        for chain_index, nodes_in_chain in chain_index_to_node.items():
            for node_in_chain_index, service_name in enumerate(nodes_in_chain):
                node_columns = get_node_columns(node_in_chain_index, with_start_time=self.gen_t_config.start_time_with_metadata)
                chain_index_to_column[chain_index][node_in_chain_index] = service_name, node_columns
        comparison_data = defaultdict(list)
        for node_in_chain_index in range(0 if is_root else 1, self.gen_t_config.chain_length):
            node_columns = get_node_columns(node_in_chain_index, with_start_time=self.gen_t_config.start_time_with_metadata)

            current_dataset = dataset[node_columns]
            metadata_columns = [c.replace(f'_{node_in_chain_index}', '', 1) for c in node_columns]
            current_dataset.columns = metadata_columns

            for chain_index in chain_column.unique():
                if node_in_chain_index in chain_index_to_column[chain_index]:
                    service_name, node_columns = chain_index_to_column[chain_index][node_in_chain_index]
                    comparison_data[service_name].append(current_dataset[chain_column == chain_index])
        self.comparison_data = {service_name: pd.concat(final_data, ignore_index=True)
                                for service_name, final_data in comparison_data.items()}

    def train_root(self):
        print("Metadata root generator started training")
        dataset, graph_column, chain_column, tx_start_time, str_columns, n_nodes, graph_index_to_edges = self.prepare()
        self.root_generator = self.root_generator or CTGAN(
            epochs=self.n_epochs, verbose=True, device=device, with_gcn=self.gen_t_config.with_gcn,
            generator_dim=self.gen_t_config.generator_dim,
            discriminator_dim=self.gen_t_config.discriminator_dim,
            n_nodes=n_nodes, graph_index_to_edges=graph_index_to_edges,
            functional_loss=partial(self.functional_loss, is_root=True),
            functional_loss_freq=self.functional_loss_freq,
            generator_lr=2e-2,
            generator_decay=1e-6,
            discriminator_lr=2e-2,
            discriminator_decay=1e-6,
            name='root',
        )

        relevant_indexes = (dataset["is_root_chain"] == True)
        graph_column = graph_column[relevant_indexes]
        chain_column = chain_column[relevant_indexes]
        tx_start_time = tx_start_time[relevant_indexes]
        root_dataset = dataset[relevant_indexes]
        self.prepare_comparison_data(root_dataset, chain_column=chain_column, is_root=True)
        print("Root dataset size:", len(root_dataset))
        if self.root_training_mid_data is None:
            self.root_training_mid_data = self.root_generator.fit(
                train_data=root_dataset,
                graph_data=graph_column,
                chain_data=chain_column,
                tx_start_time=tx_start_time,
                discrete_columns=str_columns
            )
        else:
            self.root_generator.continue_fit(
                train_data=root_dataset,
                graph_data=graph_column,
                chain_data=chain_column,
                tx_start_time=tx_start_time,
                metadata=None,
                **self.root_training_mid_data
            )
        self.use_best(is_root=True)
        self.find_best_seed(is_root=True)

    def train_chained(self):
        print("Metadata chain generator started training")
        dataset, graph_column, chain_column, tx_start_time, str_columns, n_nodes, graph_index_to_edges = self.prepare()
        self.chained_generator = self.chained_generator or CTGAN(
            epochs=self.n_epochs, verbose=True, device=device, with_gcn=self.gen_t_config.with_gcn,
            generator_dim=self.gen_t_config.generator_dim,
            discriminator_dim=self.gen_t_config.discriminator_dim,
            n_nodes=n_nodes, graph_index_to_edges=graph_index_to_edges,
            functional_loss=partial(self.functional_loss, is_root=False),
            functional_loss_freq=self.functional_loss_freq,
            generator_lr=2e-2,
            generator_decay=1e-6,
            discriminator_lr=2e-2,
            discriminator_decay=1e-6,
            name='chained',
        )

        relevant_indexes = (dataset["is_root_chain"] == False)
        graph_column = graph_column[relevant_indexes]
        chain_column = chain_column[relevant_indexes]
        tx_start_time = tx_start_time[relevant_indexes]
        chained_dataset = dataset[relevant_indexes]
        self.prepare_comparison_data(chained_dataset, chain_column=chain_column, is_root=False)

        chain_root_columns = get_node_columns(0)
        metadata = to_metadata_normalized_columns(chained_dataset[chain_root_columns], index=0)
        if self.gen_t_config.independent_chains:
            # ablation test
            metadata = pd.DataFrame(0, index=np.arange(len(metadata)), columns=metadata.columns)
        chained_dataset = chained_dataset.drop(columns=chain_root_columns)
        print("Chained dataset size:", len(chained_dataset))

        if self.chain_training_mid_data is None:
            metadata_str_columns = [c.replace('_0', '', 1) for c in set(str_columns).intersection(chain_root_columns)]
            str_columns = list(set(str_columns).difference(chain_root_columns))
            self.chain_training_mid_data = self.chained_generator.fit(
                train_data=chained_dataset,
                graph_data=graph_column,
                chain_data=chain_column,
                tx_start_time=tx_start_time,
                metadata=metadata,
                discrete_columns=str_columns,
                metadata_discrete_columns=metadata_str_columns,
            )
        else:
            self.chained_generator.continue_fit(
                train_data=chained_dataset,
                graph_data=graph_column,
                chain_data=chain_column,
                tx_start_time=tx_start_time,
                metadata=metadata,
                **self.chain_training_mid_data
            )
        self.use_best(is_root=False)
        self.find_best_seed(is_root=False)

    def _save_generator(self, gen: CTGAN, name: str):
        path = self.gen_t_config.output_dir
        pickle.dump(gen, open(f"{path}/{name}_all.pkl", "wb"))
        # This is a hack to make the model smaller
        sampler, gen._data_sampler = gen._data_sampler, DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
        gen._noise.graph_data, gen._noise.chain_data, gen._noise.trigger_data, gen._noise.tx_start_time = None, None, None, None
        gen.__doc__ = None
        gen.functional_loss = None
        gen.save(f"{path}/{name}_ctgan_generator.pkl")
        gen._data_sampler = sampler

    def save(self):
        self.save_root()
        self.save_chained()

    def save_root(self):
        path = self.gen_t_config.output_dir
        os.makedirs(path, exist_ok=True)
        self._save_generator(self.root_generator, "root")
        pickle.dump(self.column_to_values, open(f"{path}/column_to_values.pkl", "wb"))
        pickle.dump(self.root_generator.graph_index_to_edges, open(f"{path}/graph_index_to_edges.pkl", "wb"))
        pickle.dump(self.node_to_index, open(f"{path}/node_to_index.pkl", "wb"))
        pickle.dump(self.graph_index_to_chains, open(f"{path}/graph_index_to_chains.pkl", "wb"))
        pickle.dump(self.best_root_seed, open(f"{path}/best_root_seed.pkl", "wb"))
        self.save_root_local()

    def save_root_local(self):
        path = self.gen_t_config.output_dir
        pickle.dump(self.root_training_mid_data, open(f"{path}/root_local.pkl", "wb"))

    def save_chained(self):
        path = self.gen_t_config.output_dir
        os.makedirs(path, exist_ok=True)
        self._save_generator(self.chained_generator, "chained")
        pickle.dump(self.best_chained_seed, open(f"{path}/best_chained_seed.pkl", "wb"))
        self.save_chain_local()

    def save_chain_local(self):
        path = self.gen_t_config.output_dir
        pickle.dump(self.chain_training_mid_data, open(f"{path}/chain_local.pkl", "wb"))

    def load(self, only_root: bool = False, only_chained: bool = False):
        path = self.gen_t_config.output_dir
        def load_generator(name):
            generator = CTGAN.load(f"{path}/{name}_ctgan_generator.pkl")
            generator._device = device
            generator._noise.device = device
            generator._data_sampler = DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
            generator._data_sampler._n_categories = 15
            generator._data_sampler._discrete_column_matrix_st = np.zeros((1,), dtype='int')
            generator.graph_index_to_edges = pickle.load(open(f"{path}/graph_index_to_edges.pkl", "rb"))
            return generator

        if not only_chained:
            self.root_generator = load_generator("root")
            self.best_root_seed = pickle.load(open(f"{path}/best_root_seed.pkl", "rb"))
        if not only_root:
            self.chained_generator = load_generator("chained")
            self.best_chained_seed = pickle.load(open(f"{path}/best_chained_seed.pkl", "rb"))
        self.column_to_values = pickle.load(open(f"{path}/column_to_values.pkl", "rb"))
        self.node_to_index = pickle.load(open(f"{path}/node_to_index.pkl", "rb"))
        self.graph_index_to_chains = pickle.load(open(f"{path}/graph_index_to_chains.pkl", "rb"))

    def load_all(self) -> "MetadataGenerator":
        self.load_all_root()
        self.load_all_chained()
        return self

    def load_all_root(self) -> "MetadataGenerator":
        path = self.gen_t_config.output_dir
        self.load(path, only_root=True)
        self.root_generator = pickle.load(open(f"{path}/root_all.pkl", "rb"))
        self.root_generator.functional_loss = partial(self.functional_loss, is_root=True)
        self.root_training_mid_data = pickle.load(open(f"{path}/root_local.pkl", 'rb'))
        return self

    def load_all_chained(self) -> "MetadataGenerator":
        path = self.gen_t_config.output_dir
        self.load(path, only_chained=True)
        self.chained_generator = pickle.load(open(f"{path}/chained_all.pkl", "rb"))
        self.chained_generator.functional_loss = partial(self.functional_loss, is_root=False)
        self.chain_training_mid_data = pickle.load(open(f"{path}/chain_local.pkl", 'rb'))
        return self

    def _component_data_to_tx(self, graph_index: int, component_data: Dict[str, dict], tx_start_time: int, index_to_node: Dict[int, str]) -> Optional[str]:
        for node in component_data:
            component_data[node]['componentName'] = node
            if not self.gen_t_config.start_time_with_metadata:
                component_data[node]['txStartTime'] = tx_start_time
            component_data[node]['parentComponentName'] = "top"
            metadata = {}
            for key, value in component_data[node].items():
                if 'metadata' in key:
                    new_key_name = get_key_name(node, int(key.replace('metadata_', '')),
                                                config=self.gen_t_config)
                    metadata[new_key_name] = get_key_value(node, new_key_name, value, self.gen_t_config)
            component_data[node]['metadata'] = metadata
        graph_str = edge_index_to_graph(self.root_generator.graph_index_to_edges[graph_index], index_to_node)
        for source, target in eval(graph_str):
            if target in component_data:
                component_data[target]['parentComponentName'] = source

        components = prepare_components(None, config=self.gen_t_config,
                                        extracted_component_data=component_data)
        tx = prepare_tx_structure(uuid.uuid1().hex, components)
        return json.dumps(tx, cls=NpEncoder)

    def generate_traces_corpus(
            self, target_dir_path: Union[str, Path], ts_corpus: Dict[str, List[int]],
            max_occurrences_to_generate: int = 20_000,
    ):
        index_to_node = {index: node for node, index in self.node_to_index.items()}
        graph_to_index: Dict[str, int] = {edge_index_to_graph(graph, index_to_node): index
                          for index, graph in self.root_generator.graph_index_to_edges.items()}
        graph_counts = get_graph_counts(tx_start=self.gen_t_config.tx_start, tx_end=self.gen_t_config.tx_end, traces_dir=self.gen_t_config.traces_dir)

        # map graph index and time to remaining chains and generated data
        chains_to_generate: Dict[Tuple[int, int], Set[int]] = {}
        root_chains_to_generate: Dict[Tuple[int, int], Set[int]] = {}
        generated_graphs: Dict[Tuple[int, int], Dict[str, dict]] = defaultdict(dict)
        for graph, occurrences in graph_counts.items():
            if occurrences > max_occurrences_to_generate:
                print(f"Truncated {occurrences} occurrences to {max_occurrences_to_generate}")
                occurrences = max_occurrences_to_generate
            for occurrence_index in range(occurrences):
                graph_index = graph_to_index[graph]
                root_chains, chained_chains = self.graph_index_to_chains[graph_index]
                ts = ts_corpus[graph][occurrence_index] if ts_corpus else occurrence_index
                root_chains_to_generate[graph_index, ts] = set(root_chains)
                chains_to_generate[graph_index, ts] = set(chained_chains)
                generated_graphs[graph_index, ts] = {}

        def generate_bulk(is_root: bool, curr_chain_to_generate: List[Tuple[int, int, int, Optional[dict]]]):
            metadata: Optional[pd.DataFrame]
            if is_root:
                metadata = None
            else:
                trigger_fields = [c.replace(f'_{0}', '', 1) for c in get_node_columns(0)]
                metadata = pd.DataFrame([metadata for graph_index, ts, chain_index, metadata in curr_chain_to_generate])[trigger_fields]

            tx_start_time_list = pd.DataFrame([ts for graph_index, ts, chain_index, metadata in curr_chain_to_generate], columns=['txStartTime'])
            bulk_data = (self.root_generator if is_root else self.chained_generator).sample(
                graph_index_list=torch.Tensor([graph_index for graph_index, ts, chain_index, metadata in curr_chain_to_generate]),
                tx_start_time_list=tx_start_time_list,
                chain_index_list=torch.Tensor([chain_index for graph_index, ts, chain_index, metadata in curr_chain_to_generate]),
                metadata_list=metadata,
                columns=None,
                normalize_trigger_data=True,
            )

            if len(bulk_data) != len(curr_chain_to_generate):
                raise Exception(f"Expected {len(curr_chain_to_generate)} samples but got {len(bulk_data)}")
            for row, (graph_index, ts, chain_index, _) in zip(iter(bulk_data.iloc), curr_chain_to_generate):
                graph_nodes = self.column_to_values["chain"][chain_index].split('#')
                nodes_enumeration = enumerate(graph_nodes) if is_root else enumerate(graph_nodes[1:], start=1)
                for node_index, service_name in nodes_enumeration:
                    columns = get_node_columns(node_index, with_start_time=self.gen_t_config.start_time_with_metadata)
                    generated_graphs[graph_index, ts][service_name] = {
                        c.replace(f'_{node_index}', '', 1): value
                        for c, value in row[columns].items()
                    }

        # generate the root chains
        curr_chain_to_generate: List[Tuple[int, int, int, Optional[dict]]] = []
        for (graph_index, ts), chains in root_chains_to_generate.items():
            for chain in chains:
                curr_chain_to_generate.append((graph_index, ts, chain, None))
        generate_bulk(is_root=True, curr_chain_to_generate=curr_chain_to_generate)
        print("Generated root chains")

        # generate the chained chains
        while any(chains_to_generate.values()):
            curr_chain_to_generate: List[Tuple[int, int, int, Optional[dict]]] = []
            for (graph_index, ts), chains in chains_to_generate.items():
                if len(chains) > 0:
                    any_advancement = False
                    for chain_index in chains.copy():
                        row_nodes = self.column_to_values["chain"][chain_index].split('#')
                        if row_nodes[0] in generated_graphs[graph_index, ts]:
                            curr_chain_to_generate.append((graph_index, ts, chain_index, generated_graphs[graph_index, ts][row_nodes[0]]))
                            chains.remove(chain_index)
                            any_advancement = True
                    if not any_advancement:
                        raise Exception(f"Could not generate any chain for graph {graph_index} - no trigger was found")
            generate_bulk(is_root=False, curr_chain_to_generate=curr_chain_to_generate)

        os.makedirs(target_dir_path, exist_ok=True)
        output_file = open(os.path.join(target_dir_path, "generated.json"), "w")
        for (graph_index, ts), data in generated_graphs.items():
            tx = self._component_data_to_tx(graph_index, data, tx_start_time=ts, index_to_node=index_to_node)
            if tx:
                output_file.write(tx + ",\n")

    @staticmethod
    def get(gen_t_config: GenTConfig, is_roll: bool = False) -> "MetadataGenerator":
        return MetadataGenerator(
            gen_t_config,
            is_roll=is_roll,
            functional_loss_freq=10,
            functional_loss_iterations=1,
            functional_loss_cliff=min(100, gen_t_config.iterations // 2)
        )

    def functional_loss(self, iteration_index: int, is_root: bool = True):
        if iteration_index < self.functional_loss_cliff:
            return
        for repeat_index in range(self.functional_loss_iterations):
            seed = int(random.random() * (2 ** 32))
            fidelity = self.compare(sample_size=499, seed=seed, is_root=is_root)
            if fidelity < self.best_fidelity:
                gen = self.root_generator if is_root else self.chained_generator
                self.best = deepcopy(gen._generator.state_dict())
                self.best_noise = deepcopy(gen._noise.state_dict())
                self.best_fidelity = fidelity
                if is_root:
                    self.best_root_seed = seed
                else:
                    self.best_chained_seed = seed
                print(f"Best loss: ({iteration_index}/{repeat_index}): {fidelity} ({seed})", end=", ")

    def compare(self, sample_size: int, seed: Optional[int] = None, is_root: bool = True) -> float:
        """
        This function is here to support multiprocessing
        """
        seed = seed or (self.best_root_seed if is_root else self.best_chained_seed)
        generated_corpus, rows = self._generate_corpus(sample_size=sample_size, seed=seed, is_root=is_root)
        # Compare the generated corpus to the original corpus
        distances = []
        for service_name in generated_corpus['serviceName'].unique():
            generated_service = generated_corpus[generated_corpus['serviceName'] == service_name]
            original_service = self.comparison_data[service_name]
            if len(original_service) == 0 or len(generated_service) == 0:
                print(f"Skipping {service_name} because this distribution is empty")
                continue
            distances.append(compare_distributions(generated_service['gapFromParent'], original_service['gapFromParent']))
            distances.append(compare_distributions(generated_service['duration'], original_service['duration']))
        return float(numpy.mean(distances))

    def _generate_corpus(self, sample_size: int, seed: int, is_root: bool = True) -> Tuple[pd.DataFrame, List[int]]:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

        sample_size = min(sample_size, sum(len(v) for v in self.comparison_data.values()))
        gen = self.root_generator if is_root else self.chained_generator
        graphs = gen._noise.graph_data.type(torch.int64)
        timestamps = gen._noise.tx_start_time.type(torch.int64)
        triggers_data = gen._noise.trigger_data if gen._noise.trigger_data is not None else None
        chain_data: torch.Tensor = gen._noise.chain_data.type(torch.int64)

        sample_size = min(sample_size, len(graphs))
        chosen_indexes = numpy.random.choice(len(graphs), size=sample_size, replace=False)
        unique_columns = ['gapFromParent_0', 'duration_0', 'gapFromParent_1', 'duration_1', 'gapFromParent_2', 'duration_2']
        rows: pd.DataFrame = gen.sample(
            graph_index_list=graphs[chosen_indexes],
            chain_index_list=chain_data[chosen_indexes],
            metadata_list=triggers_data[chosen_indexes] if triggers_data is not None else None,
            tx_start_time_list=timestamps[chosen_indexes],
            columns=unique_columns,
            normalize_trigger_data=False
        )

        for node_index in range(0 if is_root else 1, self.gen_t_config.chain_length):
            rows[f'serviceName_{node_index}'] = None
        chain_data = chain_data[chosen_indexes].cpu().numpy()
        for chain_index in set(chain_data):
            graph_nodes = self.column_to_values["chain"][chain_index].split('#')
            nodes_enumeration = enumerate(graph_nodes) if is_root else enumerate(graph_nodes[1:], start=1)
            for node_index, service_name in nodes_enumeration:
                rows.loc[chain_data == chain_index, f'serviceName_{node_index}'] = service_name
        final_data = [
            rows[[f'gapFromParent_{node_index}', f'duration_{node_index}', f'serviceName_{node_index}']].rename(
                columns={f'gapFromParent_{node_index}': f'gapFromParent', f'duration_{node_index}': f'duration', f'serviceName_{node_index}': f'serviceName'}
            ).dropna()
            for node_index in range(0 if is_root else 1, self.gen_t_config.chain_length)
            if f'gapFromParent_{node_index}' in rows.columns
        ]

        return pd.concat(final_data, ignore_index=True), chosen_indexes

    def use_best(self, is_root: bool):
        gen = self.root_generator if is_root else self.chained_generator
        if self.best:
            gen._generator.load_state_dict(self.best)
            gen._noise.load_state_dict(self.best_noise)

    def find_best_seed(self, is_root: bool):
        seed = self.best_root_seed if is_root else self.best_chained_seed
        print("Prev best fidelity:", self.best_fidelity, seed, end=", ")
        if seed:
            self.best_fidelity = self.compare(is_root=is_root, seed=seed, sample_size=499*3)
        print("Accurate best fidelity:", self.best_fidelity, seed, end=", ")
        for _ in range(min(30, self.gen_t_config.iterations)):
            print('.', end='', flush=True)
            seed = int(random.random() * (2 ** 32))
            dist = self.compare(is_root=is_root, seed=seed, sample_size=499*3)
            if dist < self.best_fidelity:
                self.best_fidelity = dist
                if is_root:
                    self.best_root_seed = seed
                else:
                    self.best_chained_seed = seed
                print("New best fidelity:", self.best_fidelity, seed, end=", ")


def train_and_save_root(gen_t_config: GenTConfig, path: Union[str, Path], is_roll: bool = False):
    """
    This function is here to support multiprocessing
    """
    gen = MetadataGenerator.get(gen_t_config, is_roll=is_roll)
    gen.train_root()
    gen.save_root(path=str(path))
    print("Done train_and_save_root fidelity:", gen.best_fidelity)


def continue_train_and_save_root(gen_t_config: GenTConfig, path: Union[str, Path], from_path: Union[str, Path]):
    """
    This function is here to support multiprocessing
    """
    gen = MetadataGenerator.get(gen_t_config, is_roll=True)
    gen.load_all_root(path=str(from_path))
    gen.train_root()
    gen.save_root(path=str(path))
    print("\nContinue root fidelity:", gen.best_fidelity, gen.best_root_seed)


def continue_train_and_save_chained(gen_t_config: GenTConfig, path: Union[str, Path], from_path: Union[str, Path]):
    """
    This function is here to support multiprocessing
    """
    gen = MetadataGenerator.get(gen_t_config, is_roll=True)
    gen.load_all_chained(path=str(from_path))
    gen.train_chained()
    gen.save_chained(path=str(path))
    print("\nContinue chained fidelity:", gen.best_fidelity, gen.best_chained_seed)


def train_and_save_chained(gen_t_config: GenTConfig, path: Union[str, Path], is_roll: bool = False):
    """
    This function is here to support multiprocessing
    """
    gen = MetadataGenerator.get(gen_t_config, is_roll=is_roll)
    gen.train_chained()
    gen.save_chained(path=str(path))
    print("Done train_and_save_chained fidelity:", gen.best_fidelity)


if __name__ == '__main__':
    if PROFILE:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

    manager = MetadataGenerator.get(
        GenTConfig(chain_length=2, iterations=3, tx_end=100),
    )
    # manager.load_all(path="/Users/saart/cmu/GenT/results/genT/chain_length=2.iterations=300.metadata_str_size=2.metadata_int_size=3.batch_size=10.is_test=False.with_gcn=True.discriminator_dim=(128,).generator_dim=(128,).start_time_with_metadata=False.independent_chains=False.tx_start=0.tx_end=23010/metadata")
    manager.train_chained()
    # manager.find_best_seed(is_root=False)
    # manager.train_root()
    # manager.save_chained()
    # manager.load()
    # print(manager.generate(graph_index=0, tx_start_time=0))
    # roll_test()

    if PROFILE:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
