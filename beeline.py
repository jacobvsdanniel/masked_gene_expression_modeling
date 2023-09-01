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


def read_csv_header(file, dialect, write_log=True):
    if write_log:
        logger.info(f"Reading the header of {file}")

    with open(file, "r", encoding="utf8", newline="") as f:
        line = f.readline()
    reader = csv.reader([line], dialect=dialect)
    header = []
    for row in reader:
        header = row
        break

    if write_log:
        fields = len(header)
        logger.info(f"Read {fields:,} fields")
    return header


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


def extract_data_from_raw_data(cell_gene_gep_file, tf_file, gene_index_seq_file, gene_logexp_seq_file):
    # GEP
    gep_data = read_csv(cell_gene_gep_file, "csv")
    gene_list, logexp_data = gep_data[0][1:], gep_data[1:]
    del gep_data
    genes = len(gene_list)
    cells = len(logexp_data)
    logger.info(f"{cells:,} cells; {genes:,} genes")

    # TF
    all_tf_list = read_lines(tf_file)
    all_tf_set = set(all_tf_list)
    del all_tf_list
    tfs = 0
    for gene in gene_list:
        if gene in all_tf_set:
            tfs += 1
    del all_tf_set
    nontfs = genes - tfs
    logger.info(f"{tfs:,} TFs; {nontfs:,} non-TFs")

    # extract express gene sequences for each cell
    gene_index_seq_data = []
    gene_logexp_seq_data = []
    min_length, max_length = float("inf"), float("-inf")

    for raw_logexp_seq in logexp_data:
        _cell_name, raw_logexp_seq = raw_logexp_seq[0], raw_logexp_seq[1:]
        gene_index_seq = []
        gene_logexp_seq = []

        for gi, logexp in enumerate(raw_logexp_seq):
            logexp = float(logexp)
            if logexp == 0:
                continue
            gene_index_seq.append(gi)
            gene_logexp_seq.append(logexp)

        if gene_index_seq:
            gene_index_seq_data.append(gene_index_seq)
            gene_logexp_seq = np.array(gene_logexp_seq, dtype=np.float32)
            gene_logexp_seq_data.append(gene_logexp_seq)
            length = len(gene_index_seq)
            min_length = min(min_length, length)
            max_length = max(max_length, length)

    logger.info(f"# expressed genes per cell in range: [{min_length:,} {max_length:,}]")

    write_csv(gene_index_seq_file, "csv", gene_index_seq_data)
    gene_logexp_seq_data = np.array(gene_logexp_seq_data, dtype=object)
    write_npy(gene_logexp_seq_file, gene_logexp_seq_data)
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


def collect_loss():
    gpu = "2080"
    seed = 42
    source_list = ["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"]
    tf_list = ["HV"]
    nontfs_list = [500, 1000]
    config_list = ["2__80__1__80"]

    header = ["source", "TF", "nonTFs", "config", "steps", "loss"]
    data = [header]

    for source in source_list:
        for tf in tf_list:
            for nontfs in nontfs_list:
                dataset = f"{source}__{tf}TF__{nontfs}nonTFs"
                for config in config_list:

                    log_dir = os.path.join("log", "beeline", dataset, "train")
                    model = f"gpu{gpu}__model_transformer__{config}__seed{seed}"
                    log_file = os.path.join(log_dir, f"{model}.txt")

                    try:
                        line = read_lines(log_file, write_log=False)[-1]
                        assert "Best test result" in line
                    except (FileNotFoundError, IndexError, AssertionError):
                        line = None
                        logger.info(f"Missing {log_file}")

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

                    data.append([source, tf, f"{nontfs}", config, f"{steps:,}", f"{loss:.2e}"])

    loss_csv_file = "converge/beeline/loss.csv"
    write_csv(loss_csv_file, "csv", data)
    return


def collect_gene_gene_weight_for_a_batch(batch):
    (
        start, end, cells, genes,
        gene_index_seq_data, weight_dir, model,
    ) = batch
    # must use float64; addition order changes results too much for float32
    count = np.zeros((genes, genes), dtype=np.float64)
    weight = np.zeros((genes, genes), dtype=np.float64)

    for cell_index in range(start, end):
        gene_index_seq = gene_index_seq_data[cell_index]
        cell_genes = len(gene_index_seq)

        cell_weight_file = os.path.join(weight_dir, "cell", model, f"{cell_index + 1}.npy")
        w = read_npy(cell_weight_file, write_log=False)
        assert w.shape == (cell_genes, cell_genes)

        for r, r_gene_index in enumerate(gene_index_seq):
            for c, c_gene_index in enumerate(gene_index_seq):
                count[r_gene_index, c_gene_index] += 1
                weight[r_gene_index, c_gene_index] += w[r, c]

    logger.info(f"Finished [{start:,} {end:,}] among {cells:,} cells")
    return count, weight


def collect_gene_gene_weight(cell_gene_gep_file, gene_index_seq_file, weight_dir, model, processes):
    # genes
    gene_list = read_csv_header(cell_gene_gep_file, "csv")
    genes = len(gene_list) - 1
    del gene_list
    logger.info(f"{genes:,} genes")

    # expressed gene index sequence for each cell
    gene_index_seq_data = read_csv(gene_index_seq_file, "csv")
    gene_index_seq_data = [
        [int(gene_index) for gene_index in gene_index_seq]
        for gene_index_seq in gene_index_seq_data
    ]
    cells = len(gene_index_seq_data)
    logger.info(f"{cells:,} cells")

    # group cells into batches
    batch_size = round(cells**0.5 / 100) * 100
    if batch_size < 100:
        batch_size = 100
    batch_list = [
        (
            start, min(start + batch_size, cells), cells, genes,
            gene_index_seq_data, weight_dir, model,
        )
        for start in range(0, cells, batch_size)
    ]
    batches = len(batch_list)
    logger.info(f"{batches:,} batches; {batch_size:,} batch_size")

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
    logger.info(f"{genes_with_sampled_pairs:,} expressed genes")
    logger.info(f"{pairs_coexpressed_in_cells:,} expressed gene pairs")

    # save counts and weights
    target_dir = os.path.join(weight_dir, "all", f"{model}")
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
    parser.add_argument("--task", type=str, choices=["extract_seq", "collect_weight"])
    parser.add_argument("--processes", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--source", type=str, choices=["hESC", "hHep", "mDC", "mESC", "mHSC-E", "mHSC-GM", "mHSC-L"])
    parser.add_argument("--tf", type=str, choices=["HV", "ALL"])
    parser.add_argument("--nontfs", type=int, choices=[500, 1000])
    arg = parser.parse_args()
    for key, value in vars(arg).items():
        if value is not None:
            logger.info(f"[{key}] {value}")

    # collect_loss()
    # return

    data_dir = os.path.join("/", "volume", "penghsuanli-genome2-nas2", "beeline")
    species = "human" if arg.source.startswith("h") else "mouse"
    dataset = f"{arg.source}__{arg.tf}TF__{arg.nontfs}nonTFs"

    raw_dir = os.path.join(data_dir, "raw", dataset)
    os.makedirs(raw_dir, exist_ok=True)
    cell_gene_gep_file = os.path.join(raw_dir, f"GEP.csv")
    tf_file = os.path.join(data_dir, "raw", f"{species}-tfs.csv")

    seq_dir = os.path.join(data_dir, "seq", dataset)
    os.makedirs(seq_dir, exist_ok=True)
    gene_index_seq_file = os.path.join(seq_dir, "gene_index_seq.csv")
    gene_logexp_seq_file = os.path.join(seq_dir, "gene_logexp_seq.npy")

    train_dir = os.path.join(data_dir, "train", dataset)
    os.makedirs(train_dir, exist_ok=True)
    test_file = os.path.join(train_dir, "test.json")
    weight_dir = os.path.join(train_dir, "weight")

    if arg.task == "extract_seq":
        extract_data_from_raw_data(cell_gene_gep_file, tf_file, gene_index_seq_file, gene_logexp_seq_file)
        # sample_test_data(gene_index_seq_file, test_file)

    elif arg.task == "collect_weight":
        collect_gene_gene_weight(cell_gene_gep_file, gene_index_seq_file, weight_dir, arg.model, arg.processes)
        # collect_gene_gene_variance(cell_file, gene_seq_index_file, weight_dir, arg.model, arg.source, arg.processes)
    return


if __name__ == "__main__":
    main()
    sys.exit()
