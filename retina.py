import os
import csv
import sys
import json
import pickle
import random
import logging
import argparse
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
# from scipy import sparse
# from scipy.stats import ranksums, spearmanr
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


class RBOByINT:
    def __init__(self, K, p):
        assert 0 < p < 1
        self.K = K
        self.p = p
        self.normalization = (1 - p) / (1 - p**K)
        return

    def get_similarity(self, gold, auto):
        # format input
        gold = gold[:self.K]
        auto = auto[:self.K]

        # create search map
        auto_value_to_index = {v: i for i, v in enumerate(auto)}

        # count overlap
        overlap = [0] * self.K
        for gi, gv in enumerate(gold):
            ai = auto_value_to_index.get(gv, None)
            if ai is not None:
                overlap[max(gi, ai)] += 1

        for i in range(1, self.K):
            overlap[i] += overlap[i - 1]

        # compute INT (percentage of intersection) at top-k
        intersection = [
            overlap[k - 1] / k
            for k in range(1, 1 + self.K)
        ]

        # Combine INTs with decaying weights
        rbo = 0
        w = self.normalization
        for int_k in intersection:
            rbo += w * int_k
            w *= self.p

        return rbo


class RBOByRank:
    def __init__(self, K, p):
        assert 0 < p < 1
        self.K = K
        self.p = p

        w = []
        w_int = (1 - p) / (1 - p ** K)  # w_int: weight of INT_k
        for k in range(1, 1 + K):
            w_k = w_int / k  # weights of INT_k are evenly split for all its k ranks (1~k)
            w.append(w_k)
            w_int *= p  # weight decay for next INT
        # Right now, w[k] is the weight of any rank (1~k) in INT_k

        # the weight of rank k are cumulated from its weights in INT_K, INT_K-1, ..., INT_k
        for r in range(K - 2, -1, -1):
            w[r] += w[r + 1]
        # Right now, w[k] is the weight of rank k

        self.w = w
        return

    def get_similarity(self, gold, auto):
        # format input
        gold = gold[:self.K]
        auto = auto[:self.K]

        # create search map
        auto_value_to_index = {v: i for i, v in enumerate(auto)}

        # search for overlaps
        rbo = 0
        for gi, gv in enumerate(gold):
            ai = auto_value_to_index.get(gv, None)
            if ai is not None:
                i = max(gi, ai)
                rbo += self.w[i]
        return rbo


def get_percentage_of_intersection(gold_ranking, auto_ranking, k):
    gold = set(gold_ranking[:k])
    auto = set(auto_ranking[:k])
    overlap = len(gold & auto) / k
    return overlap


def get_combined_percentage_of_intersection(
        gold_ranking, auto_ranking,
        k_tuple=(100, 1000, 10000, 100000),
        w_tuple=(0.050, 0.227, 0.574, 0.148),
):
    int_list = []
    int_combined = 0

    for k, w in zip(k_tuple, w_tuple):
        int_k = get_percentage_of_intersection(gold_ranking, auto_ranking, k)
        int_list.append(int_k)
        int_combined += int_k * w

    return int_list, int_combined


def get_wjs(gold_ranking, auto_ranking, gold_weight, auto_weight, k):
    # get intersection of weight indices ranked top K in both gold and auto
    gold_set = set(gold_ranking[:k])
    auto_set = set(auto_ranking[:k])
    intersection = gold_set & auto_set

    # collect min and max of weights of conjoint weight indices
    weight_min_sum = 0
    weight_max_sum = 0

    for wi in intersection:
        gold_w = gold_weight[wi]
        auto_w = auto_weight[wi]
        weight_min_sum += min(gold_w, auto_w)
        weight_max_sum += max(gold_w, auto_w)

    try:
        wjs = weight_min_sum / weight_max_sum
    except ZeroDivisionError:
        wjs = 0
    return wjs


def get_combined_wjs(
        gold_ranking, auto_ranking, gold_weight, auto_weight,
        k_tuple=(100, 1000, 10000, 100000),
        w_tuple=(0.050, 0.227, 0.574, 0.148),
):
    wjs_list = []
    wjs_combined = 0

    for k, w in zip(k_tuple, w_tuple):
        wjs_k = get_wjs(gold_ranking, auto_ranking, gold_weight, auto_weight, k)
        wjs_list.append(wjs_k)
        wjs_combined += wjs_k * w
    return wjs_list, wjs_combined


def extract_01_hvg_data(cell_gene_count_file, hvg_file, cell_gene_count_01_hvg_file):
    # matrix
    matrix_data = read_csv(cell_gene_count_file, "csv")
    gene_list, cell_data = matrix_data[0][1:], matrix_data[1:]
    cells = len(cell_data)
    genes = len(gene_list)
    logger.info(f"{cells:,} cells")
    logger.info(f"{genes:,} genes")

    # hvg
    hvg_set = set(read_lines(hvg_file))
    hvgs = len(hvg_set)
    logger.info(f"{hvgs:,} hvgs")

    # cell-gene expression statistics
    ci_to_genes = defaultdict(lambda: 0)
    gi_to_cells = defaultdict(lambda: 0)

    for ci, row in enumerate(cell_data):
        _cell, count_list = row[0], row[1:]
        assert len(count_list) == genes
        for gi, count in enumerate(count_list):
            count = int(float(count))
            if count == 0:
                continue
            ci_to_genes[ci] += 1
            gi_to_cells[gi] += 1

    # get cells and genes with enough significance
    hundredth_cells = cells * 0.01
    hundredth_genes = genes * 0.01

    ci_list = [
        ci
        for ci in range(cells)
        if ci_to_genes[ci] >= hundredth_genes
    ]
    filtered_cells = len(ci_list)
    logger.info(f"{filtered_cells:,} filtered_cells")

    gi_list = [
        gi
        for gi in range(genes)
        if gi_to_cells[gi] >= hundredth_cells
    ]
    filtered_genes = len(gi_list)
    logger.info(f"{filtered_genes:,} filtered_genes")

    gi_list = [
        gi
        for gi in gi_list
        if gene_list[gi] in hvg_set
    ]
    filtered_genes = len(gi_list)
    logger.info(f"{filtered_genes:,} filtered_hvg_genes")

    # get filtered matrix
    header = [""] + [gene_list[gi] for gi in gi_list]
    data = [
        [cell_data[ci][0]] + [
            cell_data[ci][1 + gi]
            for gi in gi_list
        ]
        for ci in ci_list
    ]

    data = [header] + data
    write_csv(cell_gene_count_01_hvg_file, "csv", data)
    return


def extract_cell_gep(cell_gene_count_file, cell_gep_file):
    data = read_csv(cell_gene_count_file, "csv")
    header, data = data[0], data[1:]
    cells = len(data)
    genes = len(header) - 1
    logger.info(f"{cells:,} cells")
    logger.info(f"{genes:,} genes")

    cell_gep_array = np.zeros((cells, genes), dtype=np.float32)

    for ci, row in enumerate(data):
        _cell, count_list = row[0], row[1:]
        assert len(count_list) == genes

        for gi, count in enumerate(count_list):
            count = float(count)
            cell_gep_array[ci, gi] = count

        ci += 1
        if ci % 1000 == 0 or ci == cells:
            logger.info(f"{ci:,}/{cells:,}")

    norm_array = cell_gep_array.sum(axis=1)
    norm_min = norm_array.min()
    norm_max = norm_array.max()
    logger.info(f"[raw, gep count by cell] min={norm_min:,} max={norm_max:,}")

    norm_array = 1e4 / norm_array
    norm_array = np.expand_dims(norm_array, 1)

    cell_gep_array = cell_gep_array * norm_array
    norm_array = cell_gep_array.sum(axis=1)
    norm_min = norm_array.min()
    norm_max = norm_array.max()
    logger.info(f"[1e4, gep count by cell] min={norm_min:,} max={norm_max:,}")

    cell_gep_array = np.log(1 + cell_gep_array)
    logger.info(f"[log, 1e4, gep count] min={cell_gep_array.min():,} max={cell_gep_array.max():,}")

    write_npy(cell_gep_file, cell_gep_array)
    return


def extract_cell_umap(cell_gep_file, cell_umap_file):
    logger.info("extract_cell_umap()...")

    import umap
    logger.info("umap imported")

    cell_gep_matrix = read_npy(cell_gep_file)
    logger.info(f"cell_gep_matrix: {cell_gep_matrix.shape} {cell_gep_matrix.dtype} {type(cell_gep_matrix)}")

    dimension = 100
    neighbors = 15
    reducer = umap.UMAP(n_components=dimension, n_neighbors=neighbors)
    cell_umap_matrix = reducer.fit_transform(cell_gep_matrix)
    logger.info(f"cell_umap_matrix: {cell_umap_matrix.shape} {cell_umap_matrix.dtype} {type(cell_umap_matrix)}")

    write_npy(cell_umap_file, cell_umap_matrix)
    return


def extract_cell_hvgpca(cell_gep_file, hvg_index_file, cell_hvgpca_file):
    logger.info("extract_cell_hvgpca()...")

    from sklearn.decomposition import PCA
    logger.info("PCA imported")

    cell_gep_matrix = read_npy(cell_gep_file)
    logger.info(f"cell_gep_matrix: {cell_gep_matrix.shape} {cell_gep_matrix.dtype} {type(cell_gep_matrix)}")

    hvg_index_array = read_npy(hvg_index_file)
    logger.info(f"hvg_index_array: {hvg_index_array.shape} {hvg_index_array.dtype} {type(hvg_index_array)}")

    cell_hvgep_matrix = cell_gep_matrix[:, hvg_index_array]
    logger.info(f"cell_hvgep_matrix: {cell_hvgep_matrix.shape} {cell_hvgep_matrix.dtype} {type(cell_hvgep_matrix)}")

    dimension = 100
    reducer = PCA(n_components=dimension)
    cell_hvgpca_matrix = reducer.fit_transform(cell_hvgep_matrix)
    logger.info(f"cell_hvgpca_matrix: {cell_hvgpca_matrix.shape} {cell_hvgpca_matrix.dtype} {type(cell_hvgpca_matrix)}")

    write_npy(cell_hvgpca_file, cell_hvgpca_matrix)
    return


def extract_cell_neighbor(cell_emb_file, cell_neighbor_file, cell_distance_file):
    from umap.umap_ import nearest_neighbors
    from scipy.sparse import load_npz

    if cell_emb_file.endswith(".npz"):
        logger.info(f"Reading {cell_emb_file}")
        cell_emb = load_npz(cell_emb_file)
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


def compare_neighbor(cell_emb1_neighbor_file, cell_emb2_neighbor_file):
    cell_emb1_neighbor = read_npy(cell_emb1_neighbor_file)
    cell_emb2_neighbor = read_npy(cell_emb2_neighbor_file)

    cells, neighbors = cell_emb1_neighbor.shape
    total = cells * neighbors
    logger.info(f"cells * neighbors = {cells:,} * {neighbors:,} = {total:,}")
    intersection = 0
    union = 0

    for emb1_neighbor_array, emb2_neighbor_array in zip(cell_emb1_neighbor, cell_emb2_neighbor):
        emb1_set = set(emb1_neighbor_array)
        emb2_set = set(emb2_neighbor_array)
        intersection += len(emb1_set & emb2_set)
        union += len(emb1_set | emb2_set)

    ratio = intersection / union
    logger.info(f"intersection / union = {intersection:,} / {union:,} = {ratio:.1%}")
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


def extract_cell_gene_list(cell_gene_count_file, cell_file, gene_file):
    data = read_csv(cell_gene_count_file, "csv")
    header, data = data[0], data[1:]

    # gene
    assert header[0] == ""
    for gene in header[1:]:
        assert gene
    write_lines(gene_file, header[1:])

    # cell
    cell_list = [row[0] for row in data]
    write_lines(cell_file, cell_list)
    return


def extract_seq_data(cell_gene_count_file, gene_seq_index_file, gene_seq_count_file):
    data = read_csv(cell_gene_count_file, "csv")
    header, data = data[0], data[1:]
    cells = len(data)
    genes = len(header) - 1
    logger.info(f"{cells:,} cells")
    logger.info(f"{genes:,} genes")

    index_data = []
    count_data = []
    nonzeros = 0
    count_sum_max, count_sum_min = float("-inf"), float("inf")
    count_len_max, count_len_min = float("-inf"), float("inf")

    for ri, row in enumerate(data):
        cell, count_list = row[0], row[1:]
        assert len(count_list) == genes

        index_seq = []
        count_seq = []

        for index, count in enumerate(count_list):
            count = int(float(count))
            if count == 0:
                continue
            index_seq.append(index)
            count_seq.append(count)
            nonzeros += 1

        index_data.append(index_seq)
        count_data.append(count_seq)

        count_sum = sum(count_seq)
        count_sum_max = max(count_sum, count_sum_max)
        count_sum_min = min(count_sum, count_sum_min)

        count_len = len(count_seq)
        count_len_max = max(count_len, count_len_max)
        count_len_min = min(count_len, count_len_min)

        ri += 1
        if ri % 1000 == 0 or ri == cells:
            logger.info(
                f"{ri:,}/{cells:,}:"
                f" count_sum=[{count_sum_min:,} {count_sum_max:,}]"
                f" count_len=[{count_len_min:,} {count_len_max:,}]"
            )

    write_csv(gene_seq_index_file, "csv", index_data)
    write_csv(gene_seq_count_file, "csv", count_data)
    return


def extract_log_exp_seq_data(seq_count_file, seq_exp_file, seq_logexp_file):
    count_data = read_csv(seq_count_file, "csv")

    count_data = [
        np.array(count_seq, dtype=np.float32)
        for count_seq in count_data
    ]
    seq_sum_max = max(count_array.sum() for count_array in count_data)
    logger.info(f"seq_gene_count_sum_max={seq_sum_max:,}")
    scale = 1e4

    exp_data = [
        count_array * (scale / count_array.sum())
        for count_array in count_data
    ]
    logger.info(f"exp data: normalized gene count to constant sum {scale:.2e} across sequences")

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
    count_sum_max = 1e4

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


def extract_hvg_index_data(gene_file, hvg_file, hvg_index_file):
    gene_list = read_lines(gene_file)
    hvg_list = read_lines(hvg_file)

    hvg_set = set(hvg_list)
    hvg_index_list = []
    for gi, gene in enumerate(gene_list):
        if gene in hvg_set:
            hvg_index_list.append(gi)
    hvg_index_list = np.array(hvg_index_list, dtype=np.int32)
    logger.info(f"hvg_index_list: {hvg_index_list.shape}")

    np.save(hvg_index_file, hvg_index_list)
    return


def extract_tf_index_data(gene_file, tf_file, tf_index_file):
    gene_list = read_lines(gene_file)
    tf_list = read_lines(tf_file)

    tf_set = set(tf_list)
    tf_index_list = []
    for gi, gene in enumerate(gene_list):
        if gene in tf_set:
            tf_index_list.append(gi)
    tf_index_list = np.array(tf_index_list, dtype=np.int32)
    logger.info(f"tf_index_list: {tf_index_list.shape}")

    np.save(tf_index_file, tf_index_list)
    return


def validate_hvg_index_data(gene_file, hvg_file, hvg_index_file):
    gene_list = read_lines(gene_file)
    hvg_list = read_lines(hvg_file)
    hvg_index_array = read_npy(hvg_index_file)

    hvg_set = set(hvg_list)
    hvg_from_index_set = {
        gene_list[hvg_index]
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


def analyze_test_data(test_file):
    test_data = read_json(test_file)
    total_masks = 0
    total_genes = 0
    for seq_index, mask_seq in test_data:
        total_masks += sum(mask_seq)
        total_genes += len(mask_seq)
    mask_ratio = total_masks / total_genes

    logger.info(f"mask_ratio = {total_masks:,} / {total_genes:,} = {mask_ratio:.1%}")
    return


def collect_gene_gene_weight_for_a_batch(batch):
    start, end, cells, genes, gene_index_seq_data, weight_dir, model = batch

    # must use float64; addition order changes results too much for float32
    count = np.zeros((genes, genes), dtype=np.float64)
    weight = np.zeros((genes, genes), dtype=np.float64)

    for cell_index in range(start, end):
        gene_index_seq = gene_index_seq_data[cell_index]
        cell_genes = len(gene_index_seq)

        cell_index += 1
        cell_weight_file = os.path.join(weight_dir, "cell", model, f"{cell_index}.npy")
        w = read_npy(cell_weight_file, write_log=False)
        assert w.shape == (cell_genes, cell_genes)

        for r, r_gene_index in enumerate(gene_index_seq):
            for c, c_gene_index in enumerate(gene_index_seq):
                count[r_gene_index, c_gene_index] += 1
                weight[r_gene_index, c_gene_index] += w[r, c]

    logger.info(f"Finished [{start:,} {end:,}] among {cells:,} cells")
    return count, weight


def collect_gene_gene_weight(gene_seq_index_file, weight_dir, model, processes):
    # gene index sequence for each cell
    gene_index_seq_data = read_csv(gene_seq_index_file, "csv")
    gene_index_seq_data = [
        [int(gene_index) for gene_index in gene_index_seq]
        for gene_index_seq in gene_index_seq_data
    ]
    cells = len(gene_index_seq_data)
    genes = 6804

    # group cells into batches
    batch_size = round(cells**0.5 / 100) * 100
    batch_list = [
        (start, min(start + batch_size, cells), cells, genes, gene_index_seq_data, weight_dir, model)
        for start in range(0, cells, batch_size)
    ]
    batches = len(batch_list)
    logger.info(f"{cells:,} cells; {batches:,} batches; batch_size={batch_size:,}")

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
    logger.info(f"In full data:")
    logger.info(f"{genes_with_sampled_pairs:,} genes co-expressed with other genes in some cells")
    logger.info(f"{pairs_coexpressed_in_cells:,} pairs of genes co-expressed in some cells")

    # save counts and weights
    target_dir = os.path.join(weight_dir, "all", model)
    os.makedirs(target_dir, exist_ok=True)
    count_file = os.path.join(target_dir, f"count.npy")
    weight_file = os.path.join(target_dir, f"weight.npy")
    write_npy(count_file, full_count)
    write_npy(weight_file, full_weight)
    return


def analyze_weight_distribution(weight_dir, model, source):
    config_prefix = "model_transformer__"
    config_suffix = "__attnlast"
    i = model.find(config_prefix)
    j = model.find(config_suffix, i)
    config = model[i + len(config_prefix):j]

    model_dir = os.path.join(weight_dir, "all", model)
    count, weight = read_average_model_weight(model_dir)
    w = weight[count > 0]

    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    def estimate_pdf(sample_list, pdf_file):
        logger.info(f"[{pdf_file}]")

        logger.info(f"  estimating gaussian_kde kernel...")
        kernel = gaussian_kde(sample_list)

        steps = 100
        logger.info(f"  sampling {steps:,} steps...")
        x_min = min(sample_list)
        x_max = max(sample_list)
        x = np.linspace(x_min, x_max, num=steps)
        y = kernel(x)

        i_with_max_y = max(range(len(x)), key=lambda i: y[i])
        x_mode = x[i_with_max_y]
        y_max = y[i_with_max_y]
        logger.info(f"  link weight mode at {x_mode:.3f}, max density={y_max:.2e}")

        logger.info("  plotting pdf...")
        plt.plot(x, y)
        plt.xlabel("link weight")
        plt.ylabel("probability density")
        plt.savefig(pdf_file)
        plt.clf()

        return x, y, x_mode, y_max

    img_dir = os.path.join("weight_pdf", f"{config}")
    os.makedirs(img_dir, exist_ok=True)

    w_file = os.path.join(img_dir, f"w_{source}.png")
    _ = estimate_pdf(w, w_file)

    multiplier_list = [100, 1000, 10000]
    for multiplier in multiplier_list:
        logger.info(f"log_multiplier={multiplier:.0e}")
        lw = np.log(w * multiplier + 1)
        lw_file = os.path.join(img_dir, f"lw{multiplier}_{source}.png")
        _ = estimate_pdf(lw, lw_file)

    w_sorted = sorted(w, reverse=True)
    # n_list = [100, 1000, 10000, 100000]
    n_list = [100]
    for n in n_list:
        wn = w_sorted[n - 1]
        logger.info(f"w[{n}]={wn:.2e}")
        wn_file = os.path.join(img_dir, f"w{n}_{source}.png")
        _ = estimate_pdf(w_sorted[:n], wn_file)
    return


def collect_gene_gene_weight_from_seed(weight_dir, model, start, end, pool):
    start_dir = os.path.join(weight_dir, "all", f"{model}__seed{start}")
    start_count_file = os.path.join(start_dir, f"count.npy")
    start_weight_file = os.path.join(start_dir, f"weight.npy")
    count_array = read_npy(start_count_file)
    weight_array = read_npy(start_weight_file)

    for si in range(start + 1, end + 1):
        seed_dir = os.path.join(weight_dir, "all", f"{model}__seed{si}")
        seed_count_file = os.path.join(seed_dir, f"count.npy")
        seed_weight_file = os.path.join(seed_dir, f"weight.npy")
        seed_count_array = read_npy(seed_count_file)
        seed_weight_array = read_npy(seed_weight_file)
        assert (count_array == seed_count_array).all()
        if pool == "sum":
            weight_array += seed_weight_array
        elif pool == "min":
            weight_array = np.minimum(weight_array, seed_weight_array)
        elif pool == "max":
            weight_array = np.maximum(weight_array, seed_weight_array)
        else:
            assert False

    # save counts and weights
    target_dir = os.path.join(weight_dir, "all", f"{model}__seed{pool}-{start}-{end}")
    os.makedirs(target_dir, exist_ok=True)

    count_file = os.path.join(target_dir, f"count.npy")
    weight_file = os.path.join(target_dir, f"weight.npy")

    write_npy(count_file, count_array)
    write_npy(weight_file, weight_array)
    return


def read_average_model_weight(model_dir, genes=3101):
    count_file = os.path.join(model_dir, "count.npy")
    weight_file = os.path.join(model_dir, "weight.npy")
    count = read_npy(count_file)
    weight = read_npy(weight_file)

    max_count = count.max()
    max_weight = weight.max()
    logger.info(f"max_count={max_count:,} max_weight={max_weight:e}")

    assert (count[genes:, :] == 0).all()
    assert (count[:, genes:] == 0).all()
    assert (weight[genes:, :] == 0).all()
    assert (weight[:, genes:] == 0).all()
    count = count[:genes, :genes]
    weight = weight[:genes, :genes]

    assert (weight[count == 0] == 0).all()
    count_one = np.copy(count)
    count_one[count_one == 0] = 1
    weight = weight / count_one

    max_average_weight = weight.max()
    logger.info(f"max_average_weight={max_average_weight:e}")

    return count, weight


def filter_weight_by_cells(count, weight, cells):
    weight = np.copy(weight)
    cell_filter = count < cells
    weight[cell_filter] = 0
    return weight


def filter_weight_by_gene_index(weight, gene_index_array):
    weight = np.copy(weight)
    weight[gene_index_array, :] = 0
    weight[:, gene_index_array] = 0
    return weight


def filter_weight_by_diagonal(weight):
    weight = np.copy(weight)
    di = np.diag_indices_from(weight)
    weight[di] = 0
    return weight


def filter_weight_by_regulator_index(weight, regulator_index_array):
    weight = np.copy(weight)
    weight[:, regulator_index_array] = 0
    return weight


def get_nonzero_top_ranking_for_weight_matrix(weight, k=100000):
    weight = weight.reshape(-1)
    weights = weight.shape[0]
    index_list = [i for i in range(weights) if weight[i] > 0]
    nonzero_weights = len(index_list)

    # randomly breaking tie to prevent misleading results
    # misleading: e.g. predict the same weights for all gene-gene pair
    weight_random = [(w, random.random()) for w in weight]
    ranking = sorted(index_list, key=lambda i: weight_random[i], reverse=True)
    ranking = ranking[:k]

    return ranking, weight, nonzero_weights


def compare_two_models(gold_weight, auto_weight, model, cell_threshold, hvg_filter, diag_filter, tf_filter):
    k = 100000
    p = 0.9999

    # rank weights
    gold_ranking, gold_weight, gold_nonzero_weights = get_nonzero_top_ranking_for_weight_matrix(gold_weight, k=k)
    auto_ranking, auto_weight, auto_nonzero_weights = get_nonzero_top_ranking_for_weight_matrix(auto_weight, k=k)

    # INT
    int_list, int_combined = get_combined_percentage_of_intersection(gold_ranking, auto_ranking)

    # RBO
    rbo_measure = RBOByINT(k, p)
    rbo = rbo_measure.get_similarity(gold_ranking, auto_ranking)

    # F1 of INT-100 and RBO
    int_100 = int_list[0]
    try:
        int_f1 = 2 * int_100 * rbo / (int_100 + rbo)
    except ZeroDivisionError:
        int_f1 = 0

    # WJS
    wjs_list, wjs_combined = get_combined_wjs(gold_ranking, auto_ranking, gold_weight, auto_weight)

    # F1 of WJS-100 and combined WJS
    wjs_100 = wjs_list[0]
    try:
        wjs_f1 = 2 * wjs_100 * wjs_combined / (wjs_100 + wjs_combined)
    except ZeroDivisionError:
        wjs_f1 = 0

    logger.info(
        f"[results] {model} cell_threshold={cell_threshold:.0%} hvg_filter={hvg_filter} diag_filter={diag_filter} tf_filter={tf_filter}"
    )
    logger.info(
        f"  [Edges]"
        f" Lukowski={gold_nonzero_weights:,}"
        f" Menon={auto_nonzero_weights:,}"
    )
    logger.info(
        f"  [INT]"
        f" 10^2={int_list[0]:.1%}"
        f" 10^3={int_list[1]:.1%}"
        f" 10^4={int_list[2]:.1%}"
        f" 10^5={int_list[3]:.1%}"
        f" combined={int_combined:.1%}"
        f" RBO={rbo:.1%}"
    )
    logger.info(
        f"  [WJS]"
        f" 10^2={wjs_list[0]:.1%}"
        f" 10^3={wjs_list[1]:.1%}"
        f" 10^4={wjs_list[2]:.1%}"
        f" 10^5={wjs_list[3]:.1%}"
        f" combined={wjs_combined:.1%}"
    )

    return (
        gold_nonzero_weights, auto_nonzero_weights,
        int_list, int_combined, rbo, int_f1,
        wjs_list, wjs_combined, wjs_f1,
    )


def compare_all_models(gold_weight_dir, auto_weight_dir, model_list, result_file):
    genes = 3101

    # prepare cells threshold filter
    assert "lukowski" in gold_weight_dir.lower()
    assert "menon" in auto_weight_dir.lower()
    gold_cells = 20101
    auto_cells = 16797
    # cell_threshold_list = [0, 0.02, 0.05, 0.10]
    # cell_threshold_list = [0, 0.02]
    cell_threshold_list = [0]

    # prepare hvg filter
    def get_non_hvg_index_array(weight_dir):
        hvg_index_array_file = os.path.join(weight_dir, "hvg_index.npy")
        hvg_index_array = read_npy(hvg_index_array_file)
        hvg_index_set = set(hvg_index_array)
        non_hvg_index_array = np.array([i for i in range(genes) if i not in hvg_index_set])
        return non_hvg_index_array

    gold_non_hvg_index_array = get_non_hvg_index_array(gold_weight_dir)
    auto_non_hvg_index_array = get_non_hvg_index_array(auto_weight_dir)
    # hvg_filter_list = ["X", "O"]
    hvg_filter_list = ["X"]

    # prepare diag filter
    diag_filter_list = ["X", "O"]
    # diag_filter_list = ["X"]

    # prepare tf filter
    def get_non_tf_index_array(weight_dir):
        tf_index_array_file = os.path.join(weight_dir, "tf_index.npy")
        tf_index_array = read_npy(tf_index_array_file)
        tf_index_set = set(tf_index_array)
        non_tf_index_array = np.array([i for i in range(genes) if i not in tf_index_set])
        return non_tf_index_array

    gold_non_tf_index_array = get_non_tf_index_array(gold_weight_dir)
    auto_non_tf_index_array = get_non_tf_index_array(auto_weight_dir)
    tf_filter_list = ["X", "O"]
    # tf_filter_list = ["O"]
    # tf_filter_list = ["X"]

    # prepare results header
    header = [
        "model",
        "cells", "hvg", "diag", "tf",  # filter
        "Lukowski", "Menon",  # edges
        "INT_100", "INT_1000", "INT_10000", "INT_100000", "INT_comb", "RBO",
        "WJS_100", "WJS_1000", "WJS_10000", "WJS_100000", "WJS_comb",
    ]
    data = [header]

    for model in model_list:
        # read average model weight
        gold_path = os.path.join(gold_weight_dir, model)
        auto_path = os.path.join(auto_weight_dir, model)
        gold_count, gold_weight = read_average_model_weight(gold_path, genes=genes)
        auto_count, auto_weight = read_average_model_weight(auto_path, genes=genes)

        for cell_threshold in cell_threshold_list:
            gold_threshold = gold_cells * cell_threshold
            auto_threshold = auto_cells * cell_threshold
            gold_weight_cell = filter_weight_by_cells(gold_count, gold_weight, gold_threshold)
            auto_weight_cell = filter_weight_by_cells(auto_count, auto_weight, auto_threshold)

            for hvg_filter in hvg_filter_list:
                if hvg_filter == "X":
                    gold_weight_cell_hvg = np.copy(gold_weight_cell)
                    auto_weight_cell_hvg = np.copy(auto_weight_cell)
                else:
                    gold_weight_cell_hvg = filter_weight_by_gene_index(gold_weight_cell, gold_non_hvg_index_array)
                    auto_weight_cell_hvg = filter_weight_by_gene_index(auto_weight_cell, auto_non_hvg_index_array)

                for diag_filter in diag_filter_list:
                    if diag_filter == "X":
                        gold_weight_cell_hvg_diag = np.copy(gold_weight_cell_hvg)
                        auto_weight_cell_hvg_diag = np.copy(auto_weight_cell_hvg)
                    else:
                        gold_weight_cell_hvg_diag = filter_weight_by_diagonal(gold_weight_cell_hvg)
                        auto_weight_cell_hvg_diag = filter_weight_by_diagonal(auto_weight_cell_hvg)

                    for tf_filter in tf_filter_list:
                        if tf_filter == "X":
                            gold_weight_cell_hvg_diag_tf = np.copy(gold_weight_cell_hvg_diag)
                            auto_weight_cell_hvg_diag_tf = np.copy(auto_weight_cell_hvg_diag)
                        else:
                            gold_weight_cell_hvg_diag_tf = filter_weight_by_regulator_index(gold_weight_cell_hvg_diag, gold_non_tf_index_array)
                            auto_weight_cell_hvg_diag_tf = filter_weight_by_regulator_index(auto_weight_cell_hvg_diag, auto_non_tf_index_array)

                        (
                            gold_nonzero_weights, auto_nonzero_weights,
                            int_list, int_combined, rbo, int_f1,
                            wjs_list, wjs_combined, wjs_f1,
                        ) = compare_two_models(
                            gold_weight_cell_hvg_diag_tf, auto_weight_cell_hvg_diag_tf,
                            model, cell_threshold, hvg_filter, diag_filter, tf_filter,
                        )

                        measure_list = [
                            *int_list, int_combined, rbo,
                            *wjs_list, wjs_combined,
                        ]
                        measure_list = [f"{measure:.1%}" for measure in measure_list]

                        data.append([
                            model,
                            f"{cell_threshold:.0%}", hvg_filter, diag_filter, tf_filter,
                            gold_nonzero_weights, auto_nonzero_weights,
                            *measure_list,
                        ])

    write_csv(result_file, "csv", data)
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


def collect_loss():
    gpu = "2080"
    src = "lukowski3101"
    seed = 42
    layers_list = [1, 2, 4]
    dim_list = [20, 30, 40, 60, 80, 100]
    heads_list = [1, 5]
    ff_list = [20, 30, 40, 60, 80, 100]

    result_dir = os.path.join("converge", "retina")
    log_dir = os.path.join("log", "retina", "train")
    header = ["layers", "dim", "heads", "ff", "steps", "loss"]
    data = [header]

    # total_configs = len(layers_list) * len(dim_list) * len(heads_list) * len(ff_list)
    result_config_list = []

    for layers in layers_list:
        for dim in dim_list:
            for heads in heads_list:
                for ff in ff_list:

                    model = f"gpu{gpu}__model_transformer__{layers}__{dim}__{heads}__{ff}__seed{seed}__{src}"
                    log_file = os.path.join(log_dir, f"{model}.txt")

                    try:
                        line = read_lines(log_file, write_log=False)[-1]
                        assert "Best test result" in line
                    except (FileNotFoundError, IndexError, AssertionError):
                        line = None
                        logger.info(f"Missing {layers}-{dim}-{heads}-{ff}")

                    if line is None:
                        steps = 0
                        loss = 9.999
                    else:
                        steps_prefix = " steps="
                        loss_prefix = " loss="
                        si = line.find(steps_prefix)
                        sj = line.find(" ", si + 1)
                        li = line.find(loss_prefix)
                        lj = line.find(" ", li + 1)
                        steps = int(line[si + len(steps_prefix):sj].replace(",", ""))
                        loss = float(line[li + len(loss_prefix):lj])
                        result_config_list.append((loss, steps, (layers, dim, heads, ff)))

                    data.append([
                        f"{layers}", f"{dim}", f"{heads}", f"{ff}",
                        f"{steps:,}", f"{loss:.3f}",
                    ])

    loss_csv_file = os.path.join(result_dir, "loss.csv")
    write_csv(loss_csv_file, "csv", data)

    # estimate loss distribution
    # x: grid points of loss
    # y: density at loss x
    from scipy.stats import gaussian_kde
    sample_list = [loss for loss, steps, config in result_config_list]
    kernel = gaussian_kde(sample_list)
    step = 0.001
    x_left = min(sample_list)
    x_right = max(sample_list) + step
    x = np.arange(x_left, x_right, step)
    y = kernel(x)
    i_with_max_y = max(range(len(x)), key=lambda i: y[i])
    x_with_max_y = x[i_with_max_y]
    logger.info(f"loss mode at {x_with_max_y:.3f}")

    # draw loss distribution
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    loss_distribution_file = os.path.join(result_dir, "loss.png")
    plt.savefig(loss_distribution_file)

    # low loss configs
    # result_config_list = sorted(result_config_list)
    # threshold = x_with_max_y
    # low_loss_configs = 0
    # attn = "last"
    # model_list = []
    #
    # for loss, steps, (layers, dim, heads, ff) in result_config_list:
    #     if loss > threshold:
    #         break
    #     logger.info(f"{layers}-{dim}-{heads}-{ff} steps={steps:,} loss={loss:.3f}")
    #     low_loss_configs += 1
    #
    #     model = f"{gpu}/model_transformer__{layers}__{dim}__{heads}__{ff}__attn{attn}__seed{seed}"
    #     model_list.append(model)
    #
    # logger.info(f"{low_loss_configs:,} configs with loss <= {x_with_max_y:.3f}")
    # model_file = "loss/model_list.txt"
    # write_lines(model_file, model_list)
    return


def add_loss_to_measure():
    loss_file = "/volume/penghsuanli-genome2-nas2/sc/loss/loss.csv"
    # measure_file = "/volume/penghsuanli-genome2-nas2/sc/measure/grid_tfy.csv"
    # combine_file = "/volume/penghsuanli-genome2-nas2/sc/measure/grid_tfy_loss.csv"
    measure_file = "/volume/penghsuanli-genome2-nas2/sc/measure/grid_tfn.csv"
    combine_file = "/volume/penghsuanli-genome2-nas2/sc/measure/grid_tfn_loss.csv"

    # model_to_loss
    loss_data = read_csv(loss_file, "csv")
    loss_header, loss_data = loss_data[0], loss_data[1:]
    assert loss_header == ["layers", "dim", "heads", "ff", "steps", "loss"]

    attn = "last"
    seed = 42
    model_to_loss = {}

    for layers, dim, heads, ff, steps, loss in loss_data:
        model = f"model_transformer__{layers}__{dim}__{heads}__{ff}__attn{attn}__seed{seed}"
        model_to_loss[model] = (steps, loss)

    # add training result to measure data
    measure_data = read_csv(measure_file, "csv")
    measure_header, measure_data = measure_data[0], measure_data[1:]
    assert measure_header == [
        "model",
        "cells", "hvg", "diag", "tf",  # filter
        "Lukowski", "Menon",  # edges
        "INT_100", "INT_1000", "INT_10000", "INT_100000", "INT_comb", "RBO",
        "WJS_100", "WJS_1000", "WJS_10000", "WJS_100000", "WJS_comb",
    ]

    combine_header = [
        "model",
        "cells", "hvg", "diag", "tf",  # filter
        "Lukowski", "Menon",  # edges
        "steps", "loss",
        "INT_100", "INT_1000", "INT_10000", "INT_100000", "INT_comb", "RBO",
        "WJS_100", "WJS_1000", "WJS_10000", "WJS_100000", "WJS_comb",
    ]
    combine_data = [combine_header]

    for measure_datum in measure_data:
        (
            model,
            cells, hvg, diag, tf,
            Lukowski, Menon,
            INT_100, INT_1000, INT_10000, INT_100000, INT_comb, RBO,
            WJS_100, WJS_1000, WJS_10000, WJS_100000, WJS_comb,
        ) = measure_datum

        steps, loss = model_to_loss[model]

        combine_datum = (
            model,
            cells, hvg, diag, tf,
            Lukowski, Menon,
            steps, loss,
            INT_100, INT_1000, INT_10000, INT_100000, INT_comb, RBO,
            WJS_100, WJS_1000, WJS_10000, WJS_100000, WJS_comb,
        )
        combine_data.append(combine_datum)

    write_csv(combine_file, "csv", combine_data)
    return


def combine_model():
    # dataset = "lukowski3101"
    dataset = "menon3101"
    path = f"/volume/penghsuanli-genome2-nas2/retina/{dataset}/train/weight/all/2080"

    m1_prefix = "model_transformer__1__100__5__100__attnlast__seed"
    m2_prefix = "model_transformer__1__100__5__100__1__T__attnlast__seed"
    m12_suffix_list = ["42", "min-40-42", "max-40-42", "sum-40-42"]

    m3_suffix_list = ["min", "max", "sum"]
    suffix_to_combination = {
        "min": np.minimum,
        "max": np.maximum,
        "sum": np.add,
    }

    for m1_suffix in m12_suffix_list:
        m1 = m1_prefix + m1_suffix
        m1_count_file = os.path.join(path, m1, "count.npy")
        m1_weight_file = os.path.join(path, m1, "weight.npy")
        m1_count = read_npy(m1_count_file)
        m1_weight = read_npy(m1_weight_file)

        for m2_suffix in m12_suffix_list:
            m2 = m2_prefix + m2_suffix
            m2_count_file = os.path.join(path, m2, "count.npy")
            m2_weight_file = os.path.join(path, m2, "weight.npy")
            m2_count = read_npy(m2_count_file)
            m2_weight = read_npy(m2_weight_file)

            assert (m1_count == m2_count).all()
            m3_count = m1_count
            m3_prefix = f"MGM-{m1_suffix}__MGMtf1-{m2_suffix}__combine-"

            for m3_suffix in m3_suffix_list:
                m3_weight = suffix_to_combination[m3_suffix](m1_weight, m2_weight)
                m3 = m3_prefix + m3_suffix

                m3_dir = os.path.join(path, "combine", m3)
                os.makedirs(m3_dir, exist_ok=True)

                m3_count_file = os.path.join(m3_dir, "count.npy")
                m3_weight_file = os.path.join(m3_dir, "weight.npy")
                write_npy(m3_count_file, m3_count)
                write_npy(m3_weight_file, m3_weight)
    return


def tmp(do_tmp_m=True, do_spearman=True):
    from scipy.stats import spearmanr

    model1 = "/volume/penghsuanli-genome2-nas2/retina/menon3101/train/weight/all/" \
             "model_transformer__1__100__5__100__tf707__attnlast__seed42"
    model2 = "/volume/penghsuanli-genome2-nas2/retina/menon3101/train/weight/all/" \
             "3090/model_transformer__1__100__5__100__tf707__attnlast__seed42"

    def get_preprocessed_weight(model):
        count = read_npy(model + "/count.npy")
        weight = read_npy(model + "/weight.npy")

        max_count = count.max()
        max_weight = weight.max()
        logger.info(f"max_count={max_count:,} max_weight={max_weight:e}")

        assert (count[3101:, :] == 0).all()
        assert (count[:, 3101:] == 0).all()
        assert (weight[3101:, :] == 0).all()
        assert (weight[:, 3101:] == 0).all()
        count = count[:3101, :3101]
        weight = weight[:3101, :3101]

        assert (weight[count == 0] == 0).all()
        count[count == 0] = 1
        weight = weight / count
        max_average_weight = weight.max()
        logger.info(f"max_average_weight={max_average_weight:e}")

        return weight

    weight1 = get_preprocessed_weight(model1)
    weight2 = get_preprocessed_weight(model2)

    max_weight_diff = np.absolute(weight1 - weight2).max()
    logger.info(f"max_weight_diff={max_weight_diff:e}")
    if max_weight_diff == 0:
        return

    if do_tmp_m:
        def get_sorted_wi(weight):
            weight = weight.reshape(-1)
            n = weight.shape[0]
            wi = sorted(range(n), key=lambda i: weight[i], reverse=True)
            return wi

        wi1 = get_sorted_wi(weight1)
        wi2 = get_sorted_wi(weight2)

        for m in [100, 1000, 10000, 100000]:
            s1 = set(wi1[:m])
            s2 = set(wi2[:m])
            s12 = s1 & s2
            logger.info(f"s1={len(s1)}; s2={len(s2)}; s12={len(s12)}")
            rank_diff_ratio = 1 - len(s12) / m
            logger.info(f"top {m:,} edges: {rank_diff_ratio:e} difference ratio")

    if do_spearman:
        w1 = weight1.reshape(-1)
        w2 = weight2.reshape(-1)
        r, p = spearmanr(w1, w2)
        logger.info(f"spearman r={r} p={p}")
    return


def tmp2(K, p):
    # c: the weight per rank in INT_k with exponential decay
    c = []
    decay = 1
    for k in range(1, 1 + K):
        c_k = decay / k
        c.append(c_k)
        decay *= p

    # w: relative weight for a rank in RBO_K
    w = [c_k for c_k in c]
    for r in range(K - 2, -1, -1):
        w[r] += w[r + 1]

    # s: cumulated sum of normalized weight from 1 to r in RBO_K
    normalization = (1 - p) / (1 - p**K)
    s = [normalization * w_k for w_k in w]
    for r in range(1, K):
        s[r] += s[r - 1]

    r = 1
    w_r = w[r - 1] / w[0]
    s_cumulate = s[r - 1]
    logger.info(f"[RBO K={K} p={p} r={r}]")
    logger.info(f"w={w_r:.2e} s_cumulate={s_cumulate:.1%}")

    split = [100, 1000, 10000, 100000]
    for i, r in enumerate(split):
        w_r = w[r - 1] / w[0]

        s_cumulate = s[r - 1]
        r_previous = None if i == 0 else split[i - 1]
        s_cumulate_previous = 0 if r_previous is None else s[r_previous - 1]

        s_range = s_cumulate - s_cumulate_previous

        logger.info(f"[RBO K={K} p={p} r={r}]")
        logger.info(f"w={w_r:.2e} s_cumulate={s_cumulate:.1%} s_range={s_range:.1%}")
    return


def tmp3():
    N = 9000000
    K = 100000
    p = 0.9999
    k_list = [100, 1000, 10000, 100000]

    gold = list(range(N))
    auto = list(range(N))

    def perturb_ranking(ranking, window_initial, window_multiply):
        wi = 0
        window_size = window_initial
        while True:
            wj = min(wi + window_size, len(ranking))
            window = ranking[wi:wj]
            random.shuffle(window)
            for wk in range(len(window)):
                ranking[wi + wk] = window[wk]
            logger.info(f"shuffled [{wi:,} {wj:,}]")

            wi += round(window_size / 3)
            if wi >= K or wi >= len(ranking):
                break
            window_size = round(window_size * window_multiply)
        return

    radius_start, radius_multiply = 10000, 1.5
    perturb_ranking(gold, radius_start, radius_multiply)
    perturb_ranking(auto, radius_start, radius_multiply)

    gold = gold[:K]
    auto = auto[:K]
    logger.info("gold and auto rankings ready")

    rbo_by_int = RBOByINT(K, p)
    rbo_by_int = rbo_by_int.get_similarity(gold, auto)
    logger.info(f"rbo_by_int={rbo_by_int:.1%}")

    rbo_by_rank = RBOByRank(K, p)
    rbo_by_rank = rbo_by_rank.get_similarity(gold, auto)
    logger.info(f"rbo_by_rank={rbo_by_rank:.1%}")

    int_list, combined_int = get_combined_percentage_of_intersection(gold, auto)
    logger.info(f"combined_int = {combined_int:.1%}")
    average_int = sum(int_list) / len(int_list)
    logger.info(f"average_int = {average_int:.1%}")

    for k, int_k in zip(k_list, int_list):
        logger.info(f"int_{k} = {int_k:.1%}")
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--processes", type=int, default=-1)
    parser.add_argument(
        "--source", type=str, default="",
        choices=["Lukowski", "Menon", "lukowski3101", "menon3101", "lukowski2000", "menon2000"],
    )

    parser.add_argument("--seed1", type=int, default=None)
    parser.add_argument("--seed2", type=int, default=None)
    parser.add_argument("--pool", type=str, default=None, choices=["sum", "min", "max"])

    parser.add_argument("--model_list_file", type=str, default="")
    parser.add_argument("--model1", type=int, default=None)
    parser.add_argument("--model2", type=int, default=None)

    parser.add_argument("--gold", type=str, default="")
    parser.add_argument("--auto", type=str, default="")
    parser.add_argument("--measure_csv_file", type=str, default="")

    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[{key}] {value}")

    data_dir = os.path.join("/", "volume", "penghsuanli-genome2-nas2", "retina", arg.source)

    raw_dir = os.path.join(data_dir, "raw")
    cell_gene_count_file = os.path.join(raw_dir, "cell_gene_count.csv")
    cell_gene_count_01_hvg_file = os.path.join(raw_dir, "cell_gene_count_01_hvg.csv")
    cell_file = os.path.join(raw_dir, "cell.txt")
    gene_file = os.path.join(raw_dir, "gene.txt")
    hvg_file = os.path.join(raw_dir, "hvg_v2.txt")
    tf_file = os.path.join(raw_dir, "TFcheckpoint_TF.txt")

    emb_dir = os.path.join(data_dir, "emb")
    cell_gep_file = os.path.join(emb_dir, "cell_gep.npy")
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

    tf_dir = os.path.join(data_dir, "tf")
    tf_index_file = os.path.join(tf_dir, "tf_index.npy")

    train_dir = os.path.join(data_dir, "train")
    test_file = os.path.join(train_dir, "test.json")
    weight_dir = os.path.join(train_dir, "weight")

    # extract_01_hvg_data(cell_gene_count_file, hvg_file, cell_gene_count_01_hvg_file)

    # extract_cell_gep(cell_gene_count_file, cell_gep_file)
    # extract_cell_umap(cell_gep_file, cell_umap_file)
    # extract_cell_hvgpca(cell_gep_file, hvg_index_file, cell_hvgpca_file)

    # extract_cell_neighbor(cell_gep_file, cell_gep_neighbor_file, cell_gep_distance_file)
    # extract_cell_neighbor(cell_umap_file, cell_umap_neighbor_file, cell_umap_distance_file)
    # extract_cell_neighbor(cell_hvgpca_file, cell_hvgpca_neighbor_file, cell_hvgpca_distance_file)

    # compare_neighbor(cell_gep_neighbor_file, cell_umap_neighbor_file)
    # compare_neighbor(cell_gep_neighbor_file, cell_hvgpca_neighbor_file)
    # compare_neighbor(cell_umap_neighbor_file, cell_hvgpca_neighbor_file)

    # analyze_cell_neighbor(cell_gep_neighbor_file, cell_gep_distance_file)
    # analyze_cell_neighbor(cell_umap_neighbor_file, cell_umap_distance_file)
    # analyze_cell_neighbor(cell_hvgpca_neighbor_file, cell_hvgpca_distance_file)

    # extract_cell_gene_list(cell_gene_count_file, cell_file, gene_file)
    # extract_seq_data(cell_gene_count_file, gene_seq_index_file, gene_seq_count_file)
    # extract_log_exp_seq_data(gene_seq_count_file, gene_seq_exp_file, gene_seq_logexp_file)
    # validate_seq_exp(gene_seq_count_file, gene_seq_exp_file, gene_seq_logexp_file)

    # extract_hvg_index_data(gene_file, hvg_file, hvg_index_file)
    # validate_hvg_index_data(gene_file, hvg_file, hvg_index_file)
    # extract_tf_index_data(gene_file, tf_file, tf_index_file)

    # sample_test_data(gene_seq_index_file, test_file)
    # analyze_test_data(test_file)

    # tmp()
    collect_loss()
    # add_loss_to_measure()

    # model_list = read_lines(arg.model_list_file)
    # model_list = model_list[arg.model1:arg.model2 + 1]
    # for model in model_list:
    #     logger.info(f"collect_gene_gene_weight: {model}")
    #     collect_gene_gene_weight(gene_seq_index_file, weight_dir, model, arg.processes)

    # collect_gene_gene_weight(gene_seq_index_file, weight_dir, arg.model, arg.processes)
    # analyze_weight_distribution(weight_dir, arg.model, arg.source)
    # collect_gene_gene_weight_from_seed(weight_dir, arg.model, arg.seed1, arg.seed2, arg.pool)
    # combine_model()

    # model_list = [
    #     "model_transformer__2__80__1__80__attnlast__seed42",
    #     "model_transformer__1__100__1__100__attnlast__seed42",
    #     "model_transformer__1__100__5__100__attnlast__seed42",
    # ]
    # model_list = read_lines("loss/model_list.txt")
    # model_list = [m[5:] for m in model_list]
    # compare_all_models(arg.gold, arg.auto, model_list, arg.measure_csv_file)

    # K = 100000
    # for p in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
    #     tmp2(K, p)
    # tmp3()

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
