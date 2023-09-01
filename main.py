import os
import csv
import sys
import json
import math
import pickle
import random
import logging
import argparse
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
# from scipy import sparse
# from scipy.stats import ranksums
# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
csv.register_dialect(
    "csv", delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True,
    escapechar=None, lineterminator="\n", skipinitialspace=False,
)


def read_lines(file, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        line_list = f.read().splitlines()

    if write_log:
        lines = len(line_list)
        logger.info(f"Read {lines:,} lines")
    return line_list


def write_lines(file, line_list, write_log=True):
    if write_log:
        lines = len(line_list)
        logger.info(f"Writing {lines:,} lines")

    with open(file, "w", encoding="utf8") as f:
        for line in line_list:
            f.write(f"{line}\n")

    if write_log:
        logger.info(f"Written to {file}")
    return


def read_json(file, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=True):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent)

    if write_log:
        logger.info(f"Written to {file}")
    return


def read_csv(file, dialect, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, dialect=dialect)
        row_list = [row for row in reader]

    if write_log:
        rows = len(row_list)
        logger.info(f"Read {rows:,} rows")
    return row_list


def write_csv(file, dialect, row_list, write_log=True):
    if write_log:
        rows = len(row_list)
        logger.info(f"Writing {rows:,} rows")

    with open(file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, dialect=dialect)
        for row in row_list:
            writer.writerow(row)

    if write_log:
        logger.info(f"Written to {file}")
    return


def read_npy(file, allow_pickle=True, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    array = np.load(file, allow_pickle=allow_pickle)

    if write_log:
        logger.info(f"Read {array.shape} {array.dtype} array")
    return array


def write_npy(file, array, write_log=True):
    if write_log:
        logger.info(f"Writing {array.shape} {array.dtype} array")

    np.save(file, array)

    if write_log:
        logger.info(f"Written to {file}")
    return


def analyze_count_data(cell_array_file, gene_array_file, count_matrix_file, cell_meta_file):
    from scipy import sparse
    import matplotlib.pyplot as plt

    # cell
    # 43,355 cells
    cell_array = np.load(cell_array_file, allow_pickle=True)
    cells = cell_array.shape[0]
    logger.info(f"{cells:,} cells")

    # gene
    # 16,656 genes
    gene_array = np.load(gene_array_file, allow_pickle=True)
    genes = gene_array.shape[0]
    logger.info(f"{genes:,} genes")

    # cell meta
    # cell meta header: ['', 'nCount_RNA', 'nFeature_RNA', 'source']
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    logger.info(f"cell meta header: {header}")

    # count
    # loaded (43355, 16656) count_array
    count_matrix = sparse.load_npz(count_matrix_file)
    count_array = np.zeros((cells, genes), dtype=np.float32)
    count_matrix.todense(out=count_array)
    logger.info(f"loaded {count_array.shape} count_array")

    # source
    # TC: 14,857 cells
    # TJ: 8,775 cells
    # WC: 11,501 cells
    # WJ: 8,222 cells
    source_to_cells = defaultdict(lambda: 0)
    for ci, (cell, total_count, expressed_genes, source) in enumerate(cell_meta):
        total_count = float(total_count)
        expressed_genes = float(expressed_genes)

        real_cell = cell_array[ci]
        real_total_count = count_array[ci].sum()
        real_expressed_genes = np.count_nonzero(count_array[ci])
        assert real_cell == cell
        assert real_total_count == total_count
        assert real_expressed_genes == expressed_genes
        source_to_cells[source] += 1

    for source, source_cells in sorted(source_to_cells.items()):
        logger.info(f"{source}: {source_cells:,} cells")

    # expression
    count_to_values = defaultdict(lambda: 0)
    split_list = [
        (0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 640),
        (640, 100000),
    ]
    dataset_source_data = {
        "expressed_genes": defaultdict(lambda: []),
    }
    for floor, ceil in split_list:
        dataset_source_data[f"count_{floor}_{ceil}"] = defaultdict(lambda: [])

    for ci, (_cell, _total_count, expressed_genes, source) in enumerate(cell_meta):
        # statistics of non-zero counts
        for count in count_matrix.getrow(ci).data:
            count_to_values[round(count)] += 1
            for floor, ceil in split_list:
                if floor < count <= ceil:
                    break
            else:
                assert False
            dataset_source_data[f"count_{floor}_{ceil}"][source].append(count)
        dataset_source_data["expressed_genes"][source].append(expressed_genes)
        if (ci + 1) % 10000 == 0 or (ci + 1) == cells:
            logger.info(f"{ci + 1}/{cells}")

    count_values_list = sorted(count_to_values.items())
    write_csv(f"count_to_vs.csv", "csv", count_values_list)

    source_list = ["TC", "TJ", "WC", "WJ"]
    for dataset, source_to_data in dataset_source_data.items():
        logger.info(f"plotting {dataset}")
        pos_list = [i + 1 for i in range(len(source_list))]
        data = [
            np.array(source_to_data[source], dtype=np.float32)
            for source in source_list
            if source_to_data[source]
        ]
        plt.violinplot(data, pos_list)
        ax = plt.gca()
        ax.set_xticks(pos_list)
        ax.set_xticklabels(source_list)
        plt.savefig(f"img/{dataset}.png")
        plt.close()

    #     plt.boxplot(
    #         [
    #             np.array(source_to_data[source], dtype=np.float32)
    #             for source in source_list
    #         ],
    #         whis=[0, 100],
    #         labels=source_list,
    #     )
    #     plt.savefig(f"img/{dataset}.png")
    #     plt.close()
    return


def extract_log_normalized_expression(count_matrix_file, logexp_matrix_file):
    from scipy import sparse

    # count_matrix: [c, g]
    count_matrix = sparse.load_npz(count_matrix_file)
    logger.info(f"count_matrix: {count_matrix.shape} {count_matrix.dtype} {type(count_matrix)}")
    logger.info(f"count_matrix: min={count_matrix.min()} max={count_matrix.max()}")
    cells, genes = count_matrix.shape
    assert cells == 43355
    assert genes == 16656

    # norm_matrix: [c, 1]
    norm_matrix = count_matrix.sum(axis=1)
    logger.info(f"norm_matrix: {norm_matrix.shape} {norm_matrix.dtype} {type(norm_matrix)}")
    norm_max = norm_matrix.max()
    norm_min = norm_matrix.min()
    logger.info(f"[1-norm] cell_count max={norm_max:,} min={norm_min:,}")

    # scale_matrix: [c, 1]
    scale_matrix = 1e4 / norm_matrix
    logger.info(f"scale_matrix: {scale_matrix.shape} {scale_matrix.dtype} {type(scale_matrix)}")

    # exp_matrix: [c, g]
    exp_matrix = count_matrix.multiply(scale_matrix).tocsr()
    logger.info(f"exp_matrix: {exp_matrix.shape} {exp_matrix.dtype} {type(exp_matrix)}")

    # validate exp_matrix
    exp_norm_matrix = exp_matrix.sum(axis=1)
    exp_norm_max = exp_norm_matrix.max()
    exp_norm_min = exp_norm_matrix.min()
    logger.info(f"[1-norm] cell_exp max={exp_norm_max:.1f} min={exp_norm_min:.1f}")

    # logexp_matrix: [c, g]
    exp_matrix.data = np.log(1 + exp_matrix.data)
    logger.info(f"logexp_matrix: {exp_matrix.shape} {exp_matrix.dtype} {type(exp_matrix)}")
    logger.info(f"logexp_matrix: min={exp_matrix.min()} max={exp_matrix.max()}")

    sparse.save_npz(logexp_matrix_file, exp_matrix)
    return


def analyze_exp_data(cell_array_file, gene_array_file, exp_matrix_file, cell_meta_file):
    from scipy import sparse
    import matplotlib.pyplot as plt

    # cell
    # 43,355 cells
    cell_array = np.load(cell_array_file, allow_pickle=True)
    cells = cell_array.shape[0]
    logger.info(f"{cells:,} cells")

    # gene
    # 16,656 genes
    gene_array = np.load(gene_array_file, allow_pickle=True)
    genes = gene_array.shape[0]
    logger.info(f"{genes:,} genes")

    # cell meta
    # cell meta header: ['', 'nCount_RNA', 'nFeature_RNA', 'source']
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    logger.info(f"cell meta header: {header}")

    # exp
    # loaded (43355, 16656) exp_array
    exp_matrix = sparse.load_npz(exp_matrix_file)
    exp_array = np.zeros((cells, genes), dtype=np.float32)
    exp_matrix.todense(out=exp_array)
    logger.info(f"loaded {exp_array.shape} exp_array")

    # cell norm
    # [1-norm] max=31,720.0078125 min=31,719.9921875
    norm_matrix = exp_matrix.sum(axis=1)
    logger.info(f"norm_matrix: {norm_matrix.shape} {norm_matrix.dtype} {type(norm_matrix)}")
    norm_max = norm_matrix.max()
    norm_min = norm_matrix.min()
    logger.info(f"[1-norm] max={norm_max:,} min={norm_min:,}")

    # source
    # TC: 14,857 cells
    # TJ:  8,775 cells
    # WC: 11,501 cells
    # WJ:  8,222 cells
    source_to_cells = defaultdict(lambda: 0)
    for ci, (cell, _total_count, _expressed_genes, source) in enumerate(cell_meta):
        real_cell = cell_array[ci]
        assert real_cell == cell
        source_to_cells[source] += 1
    for source, source_cells in sorted(source_to_cells.items()):
        logger.info(f"{source}: {source_cells:,} cells")

    # expression
    roundlogexp_to_elements = defaultdict(lambda: 0)
    dataset_source_data = {
        "expressed_genes": defaultdict(lambda: []),
        "log_exp_all": defaultdict(lambda: []),
    }
    for digit in range(10):
        dataset_source_data[f"log_exp_{digit}_{digit + 1}"] = defaultdict(lambda: [])

    for ci, (_cell, _total_count, expressed_genes, source) in enumerate(cell_meta):
        # statistics of non-zero expressions
        for exp in exp_matrix.getrow(ci).data:
            logexp = math.log(exp)
            roundlogexp_to_elements[round(logexp)] += 1
            digit = int(logexp)
            dataset_source_data[f"log_exp_{digit}_{digit + 1}"][source].append(logexp)
            dataset_source_data[f"log_exp_all"][source].append(logexp)
        dataset_source_data["expressed_genes"][source].append(expressed_genes)
        if (ci + 1) % 1000 == 0 or (ci + 1) == cells:
            logger.info(f"{ci + 1}/{cells}")

    roundlogexp_elements_list = sorted(roundlogexp_to_elements.items())
    write_csv(f"round_log_exp_to_elements.csv", "csv", roundlogexp_elements_list)

    source_list = ["TC", "TJ", "WC", "WJ"]
    for dataset, source_to_data in dataset_source_data.items():
        logger.info(f"plotting {dataset}")
        pos_list = [i + 1 for i in range(len(source_list))]
        data = [
            np.array(source_to_data[source], dtype=np.float32)
            for source in source_list
            if source_to_data[source]
        ]
        try:
            plt.violinplot(data, pos_list)
        except ValueError as e:
            logger.info(f"ValueError: {e}")
            plt.close()
            continue
        ax = plt.gca()
        ax.set_xticks(pos_list)
        ax.set_xticklabels(source_list)
        plt.savefig(f"img/{dataset}.png")
        plt.close()
    return


def extract_cell_umap(cell_gep_file, cell_umap_file):
    from scipy import sparse
    logger.info("extract_cell_umap()...")

    import umap
    logger.info("umap imported")

    cell_gep_matrix = sparse.load_npz(cell_gep_file)
    logger.info(f"cell_gep_matrix: {cell_gep_matrix.shape} {cell_gep_matrix.dtype} {type(cell_gep_matrix)}")

    dimension = 100
    neighbors = 15
    reducer = umap.UMAP(n_components=dimension, n_neighbors=neighbors)
    cell_umap_matrix = reducer.fit_transform(cell_gep_matrix)
    logger.info(f"cell_umap_matrix: {cell_umap_matrix.shape} {cell_umap_matrix.dtype} {type(cell_umap_matrix)}")

    write_npy(cell_umap_file, cell_umap_matrix)
    return


def extract_cell_hvgpca(cell_gep_file, hvg_index_file, cell_hvgpca_file):
    from scipy import sparse
    logger.info("extract_cell_hvgpca()...")

    from sklearn.decomposition import PCA
    logger.info("PCA imported")

    cell_gep_matrix = sparse.load_npz(cell_gep_file)
    logger.info(f"cell_gep_matrix: {cell_gep_matrix.shape} {cell_gep_matrix.dtype} {type(cell_gep_matrix)}")

    hvg_index_array = read_npy(hvg_index_file)
    logger.info(f"hvg_index_array: {hvg_index_array.shape} {hvg_index_array.dtype} {type(hvg_index_array)}")

    cell_hvgep_sparse = cell_gep_matrix[:, hvg_index_array]
    cell_hvgep_matrix = cell_hvgep_sparse.todense()
    logger.info(f"cell_hvgep_matrix: {cell_hvgep_matrix.shape} {cell_hvgep_matrix.dtype} {type(cell_hvgep_matrix)}")

    dimension = 100
    reducer = PCA(n_components=dimension)
    cell_hvgpca_matrix = reducer.fit_transform(cell_hvgep_matrix)
    logger.info(f"cell_hvgpca_matrix: {cell_hvgpca_matrix.shape} {cell_hvgpca_matrix.dtype} {type(cell_hvgpca_matrix)}")

    write_npy(cell_hvgpca_file, cell_hvgpca_matrix)
    return


def extract_cell_neighbor(cell_emb_file, cell_neighbor_file, cell_distance_file):
    from scipy import sparse
    from umap.umap_ import nearest_neighbors

    if cell_emb_file.endswith(".npz"):
        logger.info(f"Reading {cell_emb_file}")
        cell_emb = sparse.load_npz(cell_emb_file)
        logger.info(f"Read {cell_emb.shape} {cell_emb.dtype} array")
    else:
        cell_emb = read_npy(cell_emb_file)

    neighbors = 15

    cell_neighbor, cell_distance, _knn_search_index = nearest_neighbors(
        cell_emb, n_neighbors=neighbors, metric="euclidean",
        metric_kwds=None, angular=False, random_state=None,
    )

    write_npy(cell_neighbor_file, cell_neighbor)
    write_npy(cell_distance_file, cell_distance)
    return


def analyze_cell_neighbor(cell_neighbor_file, cell_distance_file):
    cell_neighbor = read_npy(cell_neighbor_file)
    cell_distance = read_npy(cell_distance_file)
    cells, neighbors = cell_neighbor.shape

    # neighbor
    cell_to_indegree = defaultdict(lambda: 0)
    for ni_array in cell_neighbor:
        for ni in ni_array:
            cell_to_indegree[ni] += 1
    cell_indegree = np.array(list(cell_to_indegree.values()), dtype=np.float32)
    d_min = cell_indegree.min()
    d_max = cell_indegree.max()
    d_mean = cell_indegree.mean()
    d_std = cell_indegree.std()
    logger.info(f"in-degree in [{d_min:.1f}, {d_max:.1f}]")
    logger.info(f"mean={d_mean:.1f} std={d_std:.1f}")

    # distance
    d_min = cell_distance.min()
    d_max = cell_distance.max()
    d_mean = cell_distance.mean()
    d_std = cell_distance.std()
    logger.info(f"distance in [{d_min:.1f}, {d_max:.1f}]")
    logger.info(f"mean={d_mean:.1f} std={d_std:.1f}")
    return


def compare_neighbor(cell_emb1_neighbor_file, cell_emb2_neighbor_file):
    cell_emb1_neighbor = read_npy(cell_emb1_neighbor_file)
    cell_emb2_neighbor = read_npy(cell_emb2_neighbor_file)

    cells, neighbors = cell_emb1_neighbor.shape
    total = cells * neighbors
    logger.info(f"cells * neighbors = {cells:,} * {neighbors:,} = {total:,}")
    intersection = 0

    for emb1_neighbor_array, emb2_neighbor_array in zip(cell_emb1_neighbor, cell_emb2_neighbor):
        emb1_set = set(emb1_neighbor_array)
        emb2_set = set(emb2_neighbor_array)
        intersection += len(emb1_set & emb2_set)

    dice = intersection / total
    logger.info(f"dice = {intersection:,} / {total:,} = {dice:.1%}")
    return


def extract_seq_data(count_matrix_file, gene_seq_index_file, gene_seq_count_file):
    from scipy import sparse

    count_matrix = sparse.load_npz(count_matrix_file)
    samples, genes = count_matrix.shape
    logger.info(f"{samples:,} samples")
    logger.info(f"{genes:,} genes")

    index_data = [[] for _ in range(samples)]
    count_data = [[] for _ in range(samples)]

    x_array, y_array = count_matrix.nonzero()
    nonzeros = len(x_array)
    logger.info(f"{nonzeros:,} nonzeros")

    for ni in range(nonzeros):
        x = x_array[ni]
        y = y_array[ni]
        c = count_matrix[x, y]
        c = int(c)
        index_data[x].append(y)
        count_data[x].append(c)

        ni += 1
        if ni % 1000000 == 0 or ni == nonzeros:
            l_data = [len(index_data[xx]) for xx in range(x + 1)]
            l_min = min(l_data)
            l_max = max(l_data)
            logger.info(f"{ni:,}/{nonzeros:,}: seq length from {l_min:,} to {l_max:,}")

    write_csv(gene_seq_index_file, "csv", index_data)
    write_csv(gene_seq_count_file, "csv", count_data)
    return


def validate_seq_count(count_matrix_file, gene_seq_index_file, gene_seq_count_file):
    from scipy import sparse

    count_matrix = sparse.load_npz(count_matrix_file)
    cells, genes = count_matrix.shape
    logger.info(f"{cells:,} cells")
    logger.info(f"{genes:,} genes")

    index_data = read_csv(gene_seq_index_file, "csv")
    count_data = read_csv(gene_seq_count_file, "csv")

    for cell_index, (index_list, count_list) in enumerate(zip(index_data, count_data)):
        for i, (gene_index, gene_count) in enumerate(zip(index_list, count_list)):
            gene_index = int(gene_index)
            gene_count = int(gene_count)
            matrix_count = int(count_matrix[cell_index, gene_index])
            # logger.info(f"cell {cell_index}, expressed {i}, gene {gene_index}, count {gene_count} {matrix_count}")
            # input("YOLO")
            assert gene_count == matrix_count

        cell_index += 1
        if cell_index % 1000 == 0 or cell_index == cells:
            logger.info(f"{cell_index}/{cells} cells validated")
    return


def extract_log_exp_seq_data(seq_count_file, seq_exp_file, seq_logexp_file):
    count_data = read_csv(seq_count_file, "csv")

    count_data = [
        np.array(count_seq, dtype=np.float32)
        for count_seq in count_data
    ]
    seq_sum_max = max(count_array.sum() for count_array in count_data)
    logger.info(f"seq_gene_count_sum_max={seq_sum_max:,}")

    exp_data = [
        count_array * (seq_sum_max / count_array.sum())
        for count_array in count_data
    ]
    logger.info(f"exp data: normalized gene count to constant sum across sequences")

    logexp_data = [
        np.log(exp_array)
        for exp_array in exp_data
    ]
    logger.info(f"logexp data: log_e of exp data")

    np.save(seq_exp_file, exp_data)
    np.save(seq_logexp_file, logexp_data)
    return


def validate_seq_exp(seq_count_file, seq_exp_file, seq_logexp_file):
    count_data = read_csv(seq_count_file, "csv")
    exp_data = read_npy(seq_exp_file)
    logexp_data = read_npy(seq_logexp_file)
    count_sum_max = 31720

    for seq_index, (count_list, exp_array, logexp_array) in enumerate(zip(count_data, exp_data, logexp_data)):
        count_array = np.array([float(count) for count in count_list], dtype=np.float32)
        # logger.info(f"count_array: {count_array.shape} {count_array.dtype}")
        # logger.info(f"exp_array: {exp_array.shape} {exp_array.dtype}")
        # logger.info(f"logexp_array: {logexp_array.shape} {logexp_array.dtype}")

        count_sum = count_array.sum()
        scale = count_sum_max / count_sum
        # logger.info(f"count_sum={count_sum} scale={scale}")

        for cell_index, (count, exp, logexp) in enumerate(zip(count_array, exp_array, logexp_array)):
            scale_count = count * scale
            exp_logexp = np.exp(logexp)
            try:
                assert abs(scale_count - exp)/exp < 1e-6
                assert abs(exp_logexp - exp)/exp < 1e-6
            except AssertionError:
                logger.info(f"seq_index={seq_index} cell_index={cell_index}")
                logger.info(f"scale_count={scale_count}, exp_logexp={exp_logexp}, exp={exp}")
                input("YOLO")
    return


def extract_hvg_index_data(gene_array_file, hvg_file, hvg_index_file):
    gene_array = np.load(gene_array_file, allow_pickle=True)
    hvg_list = read_lines(hvg_file)

    hvg_set = set(hvg_list)
    hvg_index_list = []
    for gi, gene in enumerate(gene_array):
        if gene in hvg_set:
            hvg_index_list.append(gi)
    hvg_index_list = np.array(hvg_index_list, dtype=np.int32)
    logger.info(f"hvg_index_list: {hvg_index_list.shape}")

    np.save(hvg_index_file, hvg_index_list)
    return


def validate_hvg_index_data(gene_array_file, hvg_file, hvg_index_file):
    gene_array = read_npy(gene_array_file)
    hvg_list = read_lines(hvg_file)
    hvg_index_array = read_npy(hvg_index_file)

    hvg_set = set(hvg_list)
    hvg_from_index_set = {
        gene_array[hvg_index]
        for hvg_index in hvg_index_array
    }

    assert len(hvg_set) == len(hvg_from_index_set) == len(hvg_list)
    assert hvg_set == hvg_from_index_set
    return


def sample_test_data(seq_file, test_file):
    seq_data = read_csv(seq_file, "csv")
    samples = len(seq_data)
    test_samples = round(samples / 10)
    test_data = random.sample(range(samples), test_samples)

    for ti, seq_index in enumerate(test_data):
        seq_length = len(seq_data[seq_index])
        masks = round(seq_length / 10)
        mask_seq = [0] * (seq_length - masks) + [1] * masks
        random.shuffle(mask_seq)
        test_data[ti] = (seq_index, mask_seq)

    write_json(test_file, test_data)
    return


def analyze_test_data(test_file, cell_meta_file):
    cell_index_to_source = {}
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    logger.info(f"cell meta header: {header}")
    for ci, (_cell, _total_count, _expressed_genes, source) in enumerate(cell_meta):
        cell_index_to_source[ci] = source

    test_data = read_json(test_file)
    source_to_count = defaultdict(lambda: 0)
    total_masks = 0
    total_genes = 0
    for seq_index, mask_seq in test_data:
        source_to_count[cell_index_to_source[seq_index]] += 1
        total_masks += sum(mask_seq)
        total_genes += len(mask_seq)
    mask_ratio = total_masks / total_genes

    logger.info(sorted(source_to_count.items()))
    logger.info(f"mask_ratio = {total_masks:,} / {total_genes:,} = {mask_ratio:.1%}")
    return


def collect_test_split_gene_gene_weight(
        cell_meta_file, gene_seq_index_file, test_file, weight_dir,
        model, weight_source, start, end,
):
    # cell index to source
    cell_index_to_source = {}
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    logger.info(f"cell meta header: {header}")
    for ci, (_cell, _total_count, _expressed_genes, source) in enumerate(cell_meta):
        cell_index_to_source[ci] = source

    # gene index sequence for each cell
    gene_index_seq_data = read_csv(gene_seq_index_file, "csv")

    # test data sorted by cell index
    raw_test_data = read_json(test_file)
    test_data = []
    source_cells = 0
    for cell_index, mask_seq in raw_test_data:
        source = cell_index_to_source[cell_index]
        gene_index_seq = gene_index_seq_data[cell_index]
        gene_index_seq = [int(gene_index) for gene_index in gene_index_seq]
        assert len(mask_seq) == len(gene_index_seq)
        test_data.append([cell_index, source, gene_index_seq])
        if weight_source in ["pool", source]:
            source_cells += 1
    test_data = sorted(test_data)
    current_cells = 0

    # gene-gene matrices
    genes = 16656
    row_count = np.zeros(genes, dtype=np.float32)
    col_count = np.zeros(genes, dtype=np.float32)
    count = np.zeros((genes, genes), dtype=np.float32)
    weight = np.zeros((genes, genes), dtype=np.float32)

    # weight
    for test_index, (cell_index, source, gene_index_seq) in enumerate(test_data):
        if weight_source not in ["pool", source]:
            continue
        current_cells += 1
        if current_cells < start:
            continue
        if current_cells > end:
            break

        cell_weight_file = os.path.join(weight_dir, "cell", model, f"{test_index}.npy")
        w = np.load(cell_weight_file)
        cell_genes = len(gene_index_seq)
        assert w.shape == (cell_genes, cell_genes)

        for r, r_gene_index in enumerate(gene_index_seq):
            for c, c_gene_index in enumerate(gene_index_seq):
                row_count[r_gene_index] += 1
                col_count[c_gene_index] += 1
                count[r_gene_index, c_gene_index] += 1
                weight[r_gene_index, c_gene_index] += w[r, c]

        logger.info(f"[{weight_source}] ({current_cells}/[{start}-{end}]/{source_cells})")

    rows = (row_count > 0).sum()
    cols = (col_count > 0).sum()
    cells = (count > 0).sum()
    logger.info(f"[{weight_source}] non-zeroes: rows={rows,} cols={cols,} cells={cells,}")

    # save counts and weights
    target_dir = os.path.join(weight_dir, "batch", model)
    os.makedirs(target_dir, exist_ok=True)
    row_count_file = os.path.join(target_dir, f"{weight_source}_{start}_{end}_row_count.npy")
    col_count_file = os.path.join(target_dir, f"{weight_source}_{start}_{end}_col_count.npy")
    count_file = os.path.join(target_dir, f"{weight_source}_{start}_{end}_count.npy")
    weight_file = os.path.join(target_dir, f"{weight_source}_{start}_{end}_weight.npy")

    np.save(row_count_file, row_count)
    logger.info(f"saved {row_count.shape} to {row_count_file}")

    np.save(col_count_file, col_count)
    logger.info(f"saved {col_count.shape} to {col_count_file}")

    np.save(count_file, count)
    logger.info(f"saved {count.shape} to {count_file}")

    np.save(weight_file, weight)
    logger.info(f"saved {weight.shape} to {weight_file}")
    return


def collect_gene_gene_weight_for_a_batch(batch):
    (
        start, end, cells, genes,
        cell_index_list, gene_index_seq_data, weight_dir, model,
    ) = batch

    # must use float64; addition order changes results too much for float32
    count = np.zeros((genes, genes), dtype=np.float64)
    weight = np.zeros((genes, genes), dtype=np.float64)

    for source_index in range(start, end):
        gene_index_seq = gene_index_seq_data[source_index]
        cell_genes = len(gene_index_seq)

        cell_index = 1 + cell_index_list[source_index]
        cell_weight_file = os.path.join(weight_dir, "cell", model, f"{cell_index}.npy")
        w = read_npy(cell_weight_file, write_log=False)
        assert w.shape == (cell_genes, cell_genes)

        for r, r_gene_index in enumerate(gene_index_seq):
            for c, c_gene_index in enumerate(gene_index_seq):
                count[r_gene_index, c_gene_index] += 1
                weight[r_gene_index, c_gene_index] += w[r, c]

    logger.info(f"Finished [{start:,} {end:,}] among {cells:,} cells")
    return count, weight


def collect_gene_gene_weight(cell_meta_file, gene_seq_index_file, weight_dir, model, source, processes):
    genes = 16656

    # cell source metadata
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    total_cells = len(cell_meta)
    logger.info(f"cell meta header: {header}")
    logger.info(f"{total_cells:,} cells from all sources")

    # gene index sequence for each cell
    all_gene_index_seq_data = read_csv(gene_seq_index_file, "csv")
    all_gene_index_seq_data = [
        [int(gene_index) for gene_index in gene_index_seq]
        for gene_index_seq in all_gene_index_seq_data
    ]
    assert len(all_gene_index_seq_data) == total_cells

    # collect cells in desired source
    cell_index_list = []
    gene_index_seq_data = []
    for ci, (_cell, _total_count, _expressed_genes, cell_source) in enumerate(cell_meta):
        if cell_source not in source:
            continue
        cell_index_list.append(ci)
        gene_index_seq_data.append(all_gene_index_seq_data[ci])
    cells = len(gene_index_seq_data)

    # group cells into batches
    batch_size = round(cells**0.5 / 100) * 100
    batch_list = [
        (
            start, min(start + batch_size, cells), cells, genes,
            cell_index_list, gene_index_seq_data, weight_dir, model,
        )
        for start in range(0, cells, batch_size)
    ]
    batches = len(batch_list)
    logger.info(f"[{source}] {cells:,} cells; {batches:,} batches; batch_size={batch_size:,}")

    # pool weights from all batches
    with Pool(processes=processes) as pool:
        # must use float64; addition order changes results too much for float32
        # both batch splitting and imap_unordered affect addition order
        full_count = np.zeros((genes, genes), dtype=np.float64)
        full_weight = np.zeros((genes, genes), dtype=np.float64)

        for batch_count, batch_weight in pool.imap_unordered(collect_gene_gene_weight_for_a_batch, batch_list):
            full_count += batch_count
            full_weight += batch_weight

    # count number of genes and pairs which have cell samples
    genes_with_sampled_pairs = (full_count.sum(axis=1) > 0).sum()
    pairs_coexpressed_in_cells = (full_count > 0).sum()
    logger.info(f"In {source} data:")
    logger.info(f"{genes_with_sampled_pairs:,} genes co-expressed with other genes in some cells")
    logger.info(f"{pairs_coexpressed_in_cells:,} pairs of genes co-expressed in some cells")

    # save counts and weights
    target_dir = os.path.join(weight_dir, "all", f"{model}__{source}")
    os.makedirs(target_dir, exist_ok=True)
    count_file = os.path.join(target_dir, f"count.npy")
    weight_file = os.path.join(target_dir, f"weight.npy")
    write_npy(count_file, full_count)
    write_npy(weight_file, full_weight)
    return


def collect_hvg_pair_weight(hvg_index_file, weight_dir, source):
    hvg_index_array = read_npy(hvg_index_file)
    hvgs = hvg_index_array.shape[0]

    all_count_file = os.path.join(weight_dir, "all", f"{source}_count.npy")
    all_weight_file = os.path.join(weight_dir, "all", f"{source}_weight.npy")

    all_count = read_npy(all_count_file)
    all_weight = read_npy(all_weight_file)

    hvg_count = np.zeros((hvgs, hvgs), dtype=np.float32)
    hvg_weight = np.zeros((hvgs, hvgs), dtype=np.float32)

    for hi, gi in enumerate(hvg_index_array):
        for hj, gj in enumerate(hvg_index_array):
            hvg_count[hi, hj] = all_count[gi, gj]
            hvg_weight[hi, hj] = all_weight[gi, gj]

    hvg_count_file = os.path.join(weight_dir, "hvg", f"{source}_count.npy")
    hvg_weight_file = os.path.join(weight_dir, "hvg", f"{source}_weight.npy")

    write_npy(hvg_count_file, hvg_count)
    write_npy(hvg_weight_file, hvg_weight)
    return


def analyze_source_weight(weight_dir, hvg_index_file):
    source_list = ["pool", "TC", "TJ", "WC", "WJ"]
    genes = 16656
    source_to_seqs = {"pool": 4336, "TC": 1512, "TJ": 825, "WC": 1165, "WJ": 834}
    hvg_index_array = read_npy(hvg_index_file)
    hvgs = hvg_index_array.shape[0]

    for source in source_list:
        count_file = os.path.join(weight_dir, "all", f"{source}_count.npy")
        weight_file = os.path.join(weight_dir, "all", f"{source}_weight.npy")

        count_matrix = read_npy(count_file)
        weight_matrix = read_npy(weight_file)

        c_to_pairs = [0] * 10
        w_to_pairs = [0] * 100
        w_to_genes = [0] * 100

        for gi in hvg_index_array:
            for gj in hvg_index_array:
                count = count_matrix[gi, gj]
                c = int(count * 10 / source_to_seqs[source])
                if c == 10:
                    c = 9
                c_to_pairs[c] += 1

                if count == 0:
                    continue

                weight = weight_matrix[gi, gj] / count
                w = int(weight * 100)
                if w == 10:
                    w = 9
                if gi == gj:
                    w_to_genes[w] += 1
                else:
                    w_to_pairs[w] += 1

        logger.info(f"[{source}] c_to_pairs={c_to_pairs} w_to_genes={w_to_genes} w_to_pairs={w_to_pairs}")
    return


def extract_hvg_weight_diff(weight_dir, hvg_index_file):
    source_list = ["TC", "TJ", "WC", "WJ"]
    genes = 16656
    source_to_seqs = {"pool": 4336, "TC": 1512, "TJ": 825, "WC": 1165, "WJ": 834}
    hvg_index_array = read_npy(hvg_index_file)
    hvgs = hvg_index_array.shape[0]
    w_diff_threshold = 0.002
    count_ratio_threshold = 0.01

    source_to_count = {}
    source_to_weight = {}

    # normalize weight by count for HVG pairs
    for source in source_list:
        count_file = os.path.join(weight_dir, "hvg", f"{source}_count.npy")
        weight_file = os.path.join(weight_dir, "hvg", f"{source}_weight.npy")

        count_matrix = read_npy(count_file)
        weight_matrix = read_npy(weight_file)

        for hi, gi in enumerate(hvg_index_array):
            for hj, gj in enumerate(hvg_index_array):
                if count_matrix[hi, hj] == 0:
                    continue
                weight_matrix[hi, hj] = weight_matrix[hi, hj] / count_matrix[hi, hj]

        source_to_count[source] = count_matrix
        source_to_weight[source] = weight_matrix

    source_source_list = [("TC", "TJ"), ("WC", "WJ"), ("TC", "WC"), ("TJ", "WJ")]
    ss_to_hhd = {ss: [] for ss in source_source_list}
    for s1, s2 in source_source_list:
        count_matrix_1 = source_to_count[s1]
        count_matrix_2 = source_to_count[s2]
        weight_matrix_1 = source_to_weight[s1]
        weight_matrix_2 = source_to_weight[s2]

        for hi, gi in enumerate(hvg_index_array):
            for hj, gj in enumerate(hvg_index_array):
                if count_matrix_1[hi, hj] < source_to_seqs[s1] * count_ratio_threshold:
                    continue
                if count_matrix_2[hi, hj] < source_to_seqs[s2] * count_ratio_threshold:
                    continue
                w1 = weight_matrix_1[hi, hj]
                w2 = weight_matrix_2[hi, hj]
                d = abs(w1 - w2)
                if d >= w_diff_threshold:
                    ss_to_hhd[(s1, s2)].append((hi, hj, d))

    for (s1, s2), hi_hj_diff_list in ss_to_hhd.items():
        hi_hj_diff_list = sorted(hi_hj_diff_list, key=lambda hhd: hhd[2], reverse=True)
        hi_hj_diff_list = [[hi, hj, f"{diff:.2e}"] for hi, hj, diff in hi_hj_diff_list]
        file = os.path.join(weight_dir, "diff", f"{s1}_{s2}.csv")
        write_csv(file, "csv", hi_hj_diff_list)
    return


def extract_all_cell_weights_of_diff_gene_pair(
        weight_dir, hvg_index_file, cell_meta_file, gene_seq_index_file, test_file, source_1, source_2,
):
    # collect indices of target genes
    diff_gene_pair_file = os.path.join(weight_dir, "diff", f"{source_1}_{source_2}.csv")
    hi_hj_diff_list = read_csv(diff_gene_pair_file, "csv")
    hvg_index_array = read_npy(hvg_index_file)
    gg_si_w = {}
    for hi, hj, d in hi_hj_diff_list:
        gi = hvg_index_array[int(hi)]
        gj = hvg_index_array[int(hj)]
        gg_si_w[(gi, gj)] = [[], []]

    # cell index to source
    cell_index_to_source = {}
    cell_meta = read_csv(cell_meta_file, "csv")
    header, cell_meta = cell_meta[0], cell_meta[1:]
    logger.info(f"cell meta header: {header}")
    for ci, (_cell, _total_count, _expressed_genes, source) in enumerate(cell_meta):
        cell_index_to_source[ci] = source

    # gene index sequence for each cell
    gene_index_seq_data = read_csv(gene_seq_index_file, "csv")

    # test data sorted by cell index
    raw_test_data = read_json(test_file)
    test_data = []
    for cell_index, mask_seq in raw_test_data:
        source = cell_index_to_source[cell_index]
        gene_index_seq = gene_index_seq_data[cell_index]
        gene_index_seq = [int(gene_index) for gene_index in gene_index_seq]
        assert len(mask_seq) == len(gene_index_seq)
        test_data.append([cell_index, source, gene_index_seq])
    test_data = sorted(test_data)
    test_cells = len(test_data)

    # weight
    cells = 0
    weights = 0
    for test_index, (cell_index, source, gene_index_seq) in enumerate(test_data):
        if source == source_1:
            si = 0
        elif source == source_2:
            si = 1
        else:
            continue
        cells += 1
        # if cells > 100: break

        cell_weight_file = os.path.join(weight_dir, "cell", f"{test_index}.npy")
        w = np.load(cell_weight_file)
        cell_genes = len(gene_index_seq)
        assert w.shape == (cell_genes, cell_genes)

        gi_to_wi = {
            gi: wi
            for wi, gi in enumerate(gene_index_seq)
        }

        for gi, gj in gg_si_w:
            if gi in gi_to_wi and gj in gi_to_wi:
                wi = gi_to_wi[gi]
                wj = gi_to_wi[gj]
                gg_si_w[(gi, gj)][si].append(w[wi, wj])
                weights += 1

        test_index += 1
        if cells % 100 == 0 or test_index == test_cells:
            logger.info(
                f"[{source_1}-{source_2}] ({test_index}/{test_cells}):"
                f" {cells:,} cells;"
                f" {weights:,} weights"
            )

    gg_si_w_file = os.path.join(weight_dir, "diff", f"{source_1}_{source_2}_all_cell_weights.pkl")
    with open(gg_si_w_file, "wb") as f:
        pickle.dump(gg_si_w, f)
    return


def extract_hvg_pair_ranksum(weight_dir, hvg_index_file, gene_array_file):
    from scipy.stats import ranksums

    source_list = ["TC", "TJ", "WC", "WJ"]
    source_to_seqs = {"pool": 4336, "TC": 1512, "TJ": 825, "WC": 1165, "WJ": 834}
    hvg_index_array = read_npy(hvg_index_file)
    gene_array = np.load(gene_array_file, allow_pickle=True)
    source_source_list = [("TC", "TJ"), ("WC", "WJ"), ("TC", "WC"), ("TJ", "WJ")]
    numerical_tolerance = 1e-5

    source_to_count = {}
    source_to_weight = {}

    # normalize weight by count for HVG pairs
    for source in source_list:
        count_file = os.path.join(weight_dir, "hvg", f"{source}_count.npy")
        weight_file = os.path.join(weight_dir, "hvg", f"{source}_weight.npy")

        count_matrix = read_npy(count_file)
        weight_matrix = read_npy(weight_file)

        for hi, gi in enumerate(hvg_index_array):
            for hj, gj in enumerate(hvg_index_array):
                if count_matrix[hi, hj] == 0:
                    continue
                weight_matrix[hi, hj] = weight_matrix[hi, hj] / count_matrix[hi, hj]

        source_to_count[source] = count_matrix
        source_to_weight[source] = weight_matrix

    for s1, s2 in source_source_list:
        gg_si_w_file = os.path.join(weight_dir, "diff", f"{s1}_{s2}_all_cell_weights.pkl")
        with open(gg_si_w_file, "rb") as f:
            gg_si_w = pickle.load(f)

        wd_file = os.path.join(weight_dir, "diff", f"{s1}_{s2}.csv")
        hi_hj_wd_list = read_csv(wd_file, "csv")
        genei_genej_wd_pv_list = []

        for hi, hj, wd in hi_hj_wd_list:
            hi = int(hi)
            hj = int(hj)
            c1 = source_to_count[s1][hi, hj]
            c2 = source_to_count[s2][hi, hj]
            w1 = source_to_weight[s1][hi, hj]
            w2 = source_to_weight[s2][hi, hj]
            gi = hvg_index_array[hi]
            gj = hvg_index_array[hj]
            gene_i = gene_array[gi]
            gene_j = gene_array[gj]
            s1_w_list = gg_si_w[(gi, gj)][0]
            s2_w_list = gg_si_w[(gi, gj)][1]
            assert len(s1_w_list) == c1
            assert len(s2_w_list) == c2
            assert abs(sum(s1_w_list) / c1 - w1) < numerical_tolerance
            assert abs(sum(s2_w_list) / c2 - w2) < numerical_tolerance
            test_statistics, p_value = ranksums(s1_w_list, s2_w_list)
            genei_genej_wd_pv_list.append([gene_i, gene_j, float(wd), p_value])

        p_to_pairs = defaultdict(lambda: 0)
        for _gene_i, _gene_j, _wd, pv in genei_genej_wd_pv_list:
            if pv < 1e-5:
                p_to_pairs[5] += 1
            if pv < 1e-3:
                p_to_pairs[3] += 1
            if pv < 1e-2:
                p_to_pairs[2] += 1
        logger.info(
            f"[{s1}-{s2}] #pairs with -logp>x {2}:{p_to_pairs[2]:,} {3}:{p_to_pairs[3]:,} {5}:{p_to_pairs[5]:,}"
        )

        pv_file = os.path.join(weight_dir, "diff", f"{s1}_{s2}_ranksum.csv")
        data = [["HVG1", "HVG2", "weight_diff", "rank_sum_p_value"]] + [
            [gene_i, gene_j, f"{wd:.2e}", f"{pv:.2e}"]
            for gene_i, gene_j, wd, pv in genei_genej_wd_pv_list
        ]
        write_csv(pv_file, "csv", data)
    return


def test_umap():
    import umap
    from numpy.random import RandomState
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from sklearn import manifold, datasets

    rng = RandomState(0)
    n_samples = 1500
    S_points, S_color = datasets.make_s_curve(n_samples, random_state=rng)

    def plot_3d(points, points_color, title):
        x, y, z = points.T

        fig, ax = plt.subplots(
            figsize=(6, 6),
            facecolor="white",
            tight_layout=True,
            subplot_kw={"projection": "3d"},
        )
        fig.suptitle(title, size=16)
        col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
        ax.view_init(azim=-60, elev=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

        fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
        plt.savefig("png/" + title.replace(" ", "_").replace("\n", "_") + ".png")
        return

    def plot_2d(points, points_color, title):
        fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
        fig.suptitle(title, size=16)
        add_2d_scatter(ax, points, points_color)
        plt.savefig("png/" + title.replace(" ", "_").replace("\n", "_") + ".png")

    def add_2d_scatter(ax, points, points_color, title=None):
        x, y = points.T
        ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        return

    n_components = 2

    # raw
    logger.info("[raw]")
    plot_3d(S_points, S_color, "raw")

    # mds
    logger.info("[mds]")
    md_scaling = manifold.MDS(n_components=n_components, max_iter=50, n_init=4, random_state=rng)
    S_scaling = md_scaling.fit_transform(S_points)
    plot_2d(S_scaling, S_color, "mds")

    # tsne
    logger.info("[tsne]")
    t_sne = manifold.TSNE(
        n_components=n_components,
        learning_rate="auto",
        perplexity=30,
        n_iter=250,
        init="random",
        random_state=rng,
    )
    S_t_sne = t_sne.fit_transform(S_points)
    plot_2d(S_t_sne, S_color, "tsne")

    # umap
    logger.info("[umap]")
    umap_reducer = umap.UMAP(n_components=n_components, n_neighbors=1500)
    S_umap = umap_reducer.fit_transform(S_points)
    plot_2d(S_umap, S_color, "umap")

    logger.info("done")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=4336)
    parser.add_argument("--processes", type=int, default=-1)
    parser.add_argument("--model", type=str)
    parser.add_argument("--source", type=str, default="TC-TJ-WC-WJ")
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        logger.info(f"[{key}] {value}")

    data_dir = os.path.join("/", "volume", "penghsuanli-genome2-nas2", "macrophage")

    raw_dir = os.path.join(data_dir, "raw")
    cell_array_file = os.path.join(raw_dir, "row_cells.npy")
    gene_array_file = os.path.join(raw_dir, "col_genes.npy")
    count_matrix_file = os.path.join(raw_dir, "RNA_sparse_count.npz")
    cell_meta_file = os.path.join(raw_dir, "metadata.csv")
    logexp_matrix_file = os.path.join(raw_dir, "sparse_logexp.npz")
    hvg_file = os.path.join(raw_dir, "hvg_v2.txt")

    emb_dir = os.path.join(data_dir, "emb")
    cell_umap_file = os.path.join(emb_dir, "cell_umap.npy")
    cell_hvgpca_file = os.path.join(emb_dir, "cell_hvgpca.npy")

    cell_gep_neighbor_file = os.path.join(emb_dir, "cell_gep_neighbor.npy")
    cell_gep_distance_file = os.path.join(emb_dir, "cell_gep_distance.npy")
    cell_umap_neighbor_file = os.path.join(emb_dir, "cell_umap_neighbor.npy")
    cell_umap_distance_file = os.path.join(emb_dir, "cell_umap_distance.npy")
    cell_hvgpca_neighbor_file = os.path.join(emb_dir, "cell_hvgpca_neighbor.npy")
    cell_hvgpca_distance_file = os.path.join(emb_dir, "cell_hvgpca_distance.npy")

    seq_dir = os.path.join(data_dir, "seq")
    gene_seq_index_file = os.path.join(seq_dir, "gene_seq_index.csv")
    gene_seq_count_file = os.path.join(seq_dir, "gene_seq_count.csv")
    gene_seq_exp_file = os.path.join(seq_dir, "gene_seq_exp.npy")
    gene_seq_logexp_file = os.path.join(seq_dir, "gene_seq_logexp.npy")

    hvg_dir = os.path.join(data_dir, "hvg")
    hvg_index_file = os.path.join(hvg_dir, "hvg_index.npy")

    train_dir = os.path.join(data_dir, "train")
    test_file = os.path.join(train_dir, "test.json")
    weight_dir = os.path.join(train_dir, "weight")

    # analyze_count_data(cell_array_file, gene_array_file, count_matrix_file, cell_meta_file)
    # extract_log_normalized_expression(count_matrix_file, logexp_matrix_file)
    # analyze_exp_data(cell_array_file, gene_array_file, exp_matrix_file, cell_meta_file)

    # extract_cell_umap(logexp_matrix_file, cell_umap_file)
    # extract_cell_hvgpca(logexp_matrix_file, hvg_index_file, cell_hvgpca_file)

    # extract_cell_neighbor(logexp_matrix_file, cell_gep_neighbor_file, cell_gep_distance_file)
    # extract_cell_neighbor(cell_umap_file, cell_umap_neighbor_file, cell_umap_distance_file)
    # extract_cell_neighbor(cell_hvgpca_file, cell_hvgpca_neighbor_file, cell_hvgpca_distance_file)

    # analyze_cell_neighbor(cell_gep_neighbor_file, cell_gep_distance_file)
    # analyze_cell_neighbor(cell_umap_neighbor_file, cell_umap_distance_file)
    # analyze_cell_neighbor(cell_hvgpca_neighbor_file, cell_hvgpca_distance_file)

    # compare_neighbor(cell_gep_neighbor_file, cell_umap_neighbor_file)
    # compare_neighbor(cell_gep_neighbor_file, cell_hvgpca_neighbor_file)
    # compare_neighbor(cell_umap_neighbor_file, cell_hvgpca_neighbor_file)

    # extract_seq_data(count_matrix_file, gene_seq_index_file, gene_seq_count_file)
    # validate_seq_count(count_matrix_file, gene_seq_index_file, gene_seq_count_file)
    # extract_log_exp_seq_data(gene_seq_count_file, gene_seq_exp_file, gene_seq_logexp_file)
    # validate_seq_exp(gene_seq_count_file, gene_seq_exp_file, gene_seq_logexp_file)

    # extract_hvg_index_data(gene_array_file, hvg_file, hvg_index_file)
    # validate_hvg_index_data(gene_array_file, hvg_file, hvg_index_file)

    # sample_test_data(gene_seq_index_file, test_file)
    # analyze_test_data(test_file, cell_meta_file)

    collect_gene_gene_weight(cell_meta_file, gene_seq_index_file, weight_dir, arg.model, arg.source, arg.processes)
    # collect_gene_gene_weight_from_batch(weight_dir, arg.model, arg.source)
    # collect_hvg_pair_weight(hvg_index_file, weight_dir, arg.source)
    # extract_hvg_weight_diff(weight_dir, hvg_index_file)
    # extract_all_cell_weights_of_diff_gene_pair(
    #     weight_dir, hvg_index_file, cell_meta_file, gene_seq_index_file, test_file, arg.s1, arg.s2,
    # )
    # extract_hvg_pair_ranksum(weight_dir, hvg_index_file, gene_array_file)
    return


if __name__ == "__main__":
    main()
    sys.exit()
