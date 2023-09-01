import os
import csv
import sys
import json
import random
import logging
import argparse
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
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


def extract_data_from_base_data(
        base_count_matrix_file, base_cell_meta_file, base_gene_file,
        cell_file, gene_file,
        gene_seq_index_file, gene_seq_count_file,
):
    from scipy import sparse

    # base data
    base_count_matrix = sparse.load_npz(base_count_matrix_file)
    logger.info(f"base_count_matrix: {base_count_matrix.shape}")
    base_count_array = np.zeros(base_count_matrix.shape, dtype=np.float32)
    base_count_matrix.todense(out=base_count_array)
    logger.info(f"created dense {base_count_array.shape} base_count_array")

    base_cell_meta = read_csv(base_cell_meta_file, "csv")
    cell_meta_header, base_cell_meta = base_cell_meta[0], base_cell_meta[1:]
    logger.info(f"cell meta header: {cell_meta_header}")  # ['', 'nCount_RNA', 'nFeature_RNA', 'source']

    base_gene_array = read_npy(base_gene_file)

    # map base data gene index to target data gene index
    gene_list = read_lines(gene_file)
    gene_to_gi = {gene: gi for gi, gene in enumerate(gene_list)}
    base_gi_to_target_gi = {
        base_gi: gene_to_gi[gene]
        for base_gi, gene in enumerate(base_gene_array)
        if gene in gene_to_gi
    }
    assert len(base_gi_to_target_gi) == len(gene_list) == 2770

    # extract target data
    cell_meta = [cell_meta_header]
    gene_seq_index_data = []
    gene_seq_count_data = []
    min_length, max_length = float("inf"), float("-inf")

    for base_ci, (cell, base_count, base_expressed_genes, source) in enumerate(base_cell_meta):
        target_count = 0
        target_expressed_genes = 0
        gene_index_seq = []
        gene_count_seq = []

        for base_gi, count in enumerate(base_count_array[base_ci]):
            if count == 0:
                continue
            if base_gi not in base_gi_to_target_gi:
                continue
            target_count += count
            target_expressed_genes += 1

            gi = base_gi_to_target_gi[base_gi]
            gene_index_seq.append(gi)
            gene_count_seq.append(count)

        if gene_index_seq:
            cell_meta.append([cell, str(target_count), str(target_expressed_genes), source])
            gene_seq_index_data.append(gene_index_seq)
            gene_seq_count_data.append(gene_count_seq)
            length = len(gene_index_seq)
            min_length = min(min_length, length)
            max_length = max(max_length, length)

    logger.info(f"# expressed target genes per cell in range: [{min_length:,} {max_length:,}]")

    write_csv(cell_file, "csv", cell_meta)
    write_csv(gene_seq_index_file, "csv", gene_seq_index_data)
    write_csv(gene_seq_count_file, "csv", gene_seq_count_data)
    return


def extract_log_exp_seq_data(seq_count_file, seq_logexp_file):
    count_data = read_csv(seq_count_file, "csv")

    count_data = [
        np.array(count_seq, dtype=np.float32)
        for count_seq in count_data
    ]
    seq_sum_max = max(count_array.sum() for count_array in count_data)
    logger.info(f"seq_gene_count_sum_max={seq_sum_max:,}")

    exp_data = [
        count_array * (10000 / count_array.sum())
        for count_array in count_data
    ]
    logger.info(f"exp data: normalized gene count to constant sum across sequences")

    logexp_data = [
        np.log(exp_array)
        for exp_array in exp_data
    ]
    logger.info(f"logexp data: log_e of exp data")

    np.save(seq_logexp_file, logexp_data)
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
    genes = 2770

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


def read_average_weight(weight_dir):
    count_file = os.path.join(weight_dir, "count.npy")
    weight_file = os.path.join(weight_dir, "weight.npy")

    count = read_npy(count_file)
    weight = read_npy(weight_file)

    assert (weight[count == 0] == 0).all()
    count_one = np.copy(count)
    count_one[count_one == 0] = 1
    weight = weight / count_one

    max_average_weight = weight.max()
    logger.info(f"max_average_weight={max_average_weight:e}")

    return count, weight


def collect_gene_gene_variance_for_a_batch(batch):
    (
        start, end, cells, genes,
        cell_index_list, gene_index_seq_data, weight_dir, model, weight,
    ) = batch

    # must use float64; addition order changes results too much for float32
    variance = np.zeros((genes, genes), dtype=np.float64)

    for source_index in range(start, end):
        gene_index_seq = gene_index_seq_data[source_index]
        cell_genes = len(gene_index_seq)

        cell_index = 1 + cell_index_list[source_index]
        cell_weight_file = os.path.join(weight_dir, "cell", model, f"{cell_index}.npy")
        w = read_npy(cell_weight_file, write_log=False)
        assert w.shape == (cell_genes, cell_genes)

        for r, r_gi in enumerate(gene_index_seq):
            for c, c_gi in enumerate(gene_index_seq):
                variance[r_gi, c_gi] += (w[r, c] - weight[r_gi, c_gi]) ** 2

    logger.info(f"Finished [{start:,} {end:,}] among {cells:,} cells")
    return variance


def get_sample_std(variance, count):
    assert (variance[count == 0] == 0).all()
    count_one = np.copy(count)
    count_one[count_one == 0] = 1
    variance = variance / count_one
    std = variance ** 0.5

    max_std = std.max()
    logger.info(f"max_std={max_std:e}")
    return std


def collect_gene_gene_variance(cell_meta_file, gene_seq_index_file, weight_dir, model, source, processes):
    genes = 2770

    # load counts and weights
    target_dir = os.path.join(weight_dir, "all", f"{model}__{source}")
    count, weight = read_average_weight(target_dir)

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
            cell_index_list, gene_index_seq_data, weight_dir, model, weight,
        )
        for start in range(0, cells, batch_size)
    ]
    batches = len(batch_list)
    logger.info(f"[{source}] {cells:,} cells; {batches:,} batches; batch_size={batch_size:,}")

    # pool weights from all batches
    with Pool(processes=processes) as pool:
        # must use float64; addition order changes results too much for float32
        # both batch splitting and imap_unordered affect addition order
        full_variance = np.zeros((genes, genes), dtype=np.float64)

        for batch_variance in pool.imap_unordered(collect_gene_gene_variance_for_a_batch, batch_list):
            full_variance += batch_variance

    # compute (biased) sample standard deviation
    _std = get_sample_std(full_variance, count)

    # write variance (not averaged by sample count)
    variance_file = os.path.join(target_dir, "variance.npy")
    write_npy(variance_file, full_variance)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--source", type=str)
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[{key}] {value}")

    base_dir = os.path.join("/", "volume", "penghsuanli-genome2-nas2", "macrophage", "raw")
    base_count_matrix_file = os.path.join(base_dir, "RNA_sparse_count.npz")
    base_cell_meta_file = os.path.join(base_dir, "metadata.csv")
    base_gene_file = os.path.join(base_dir, "col_genes.npy")

    data_dir = os.path.join("/", "volume", "penghsuanli-genome2-nas2", "brain2770")

    raw_dir = os.path.join(data_dir, "raw")
    cell_file = os.path.join(raw_dir, "metadata.csv")
    gene_file = os.path.join(raw_dir, "hvg_pool_TF.txt")

    seq_dir = os.path.join(data_dir, "seq")
    gene_seq_index_file = os.path.join(seq_dir, "gene_seq_index.csv")
    gene_seq_count_file = os.path.join(seq_dir, "gene_seq_count.csv")
    gene_seq_logexp_file = os.path.join(seq_dir, "gene_seq_logexp.npy")

    train_dir = os.path.join(data_dir, "train")
    test_file = os.path.join(train_dir, "test.json")
    weight_dir = os.path.join(train_dir, "weight")

    # extract_data_from_base_data(
    #     base_count_matrix_file, base_cell_meta_file, base_gene_file,
    #     cell_file, gene_file,
    #     gene_seq_index_file, gene_seq_count_file,
    # )
    # extract_log_exp_seq_data(gene_seq_count_file, gene_seq_logexp_file)

    # sample_test_data(gene_seq_index_file, test_file)
    # analyze_test_data(test_file, cell_file)

    # collect_gene_gene_weight(cell_file, gene_seq_index_file, weight_dir, arg.model, arg.source, arg.processes)
    collect_gene_gene_variance(cell_file, gene_seq_index_file, weight_dir, arg.model, arg.source, arg.processes)
    return


if __name__ == "__main__":
    main()
    sys.exit()
