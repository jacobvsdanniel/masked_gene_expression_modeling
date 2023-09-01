import os
import csv
import sys
import copy
import json
import random
import logging
import argparse

import torch
import numpy as np

from model import (
    GeneUtils,
    Model_autoencoder,
)

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
csv.register_dialect(
    "tsv", delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, doublequote=False,
    escapechar=None, lineterminator="\n", skipinitialspace=False,
)
mask_ratio = 0.1


def read_json(file, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


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


def read_npy(file, allow_pickle=True, write_log=True):
    if write_log:
        logger.info(f"Reading {file}")

    array = np.load(file, allow_pickle=allow_pickle)

    if write_log:
        logger.info(f"Read {array.shape} {array.dtype} array")
    return array


class Dataset:
    def __init__(self, cell_gene_file, cell_exp_file, test_file, hvg_file=None):
        cell_gene_data = read_csv(cell_gene_file, "csv")
        cell_exp_data = read_npy(cell_exp_file)
        testcell_maskseq_data = read_json(test_file)

        self.train_data = []
        self.test_data = []

        if hvg_file:
            self.hvg_set = set(read_npy(hvg_file))
            hvg_indices = len(self.hvg_set)
            logger.info(f"{hvg_indices:,} hvg_indices")
        else:
            self.hvg_set = None

        testcell_to_maskseq = dict(testcell_maskseq_data)

        for cell, (gene_seq, exp_seq) in enumerate(zip(cell_gene_data, cell_exp_data)):
            assert len(gene_seq) == len(exp_seq)
            gene_seq = [int(gene) for gene in gene_seq]
            exp_seq = exp_seq.tolist()

            if cell in testcell_to_maskseq:
                mask_seq = testcell_to_maskseq[cell]
                assert len(gene_seq) == len(mask_seq)
                hvgmask_seq = [
                    1 if mask == 1 and (self.hvg_set is None or gene in self.hvg_set) else 0
                    for gene, mask in zip(gene_seq, mask_seq)
                ]
                self.test_data.append((gene_seq, exp_seq, mask_seq, hvgmask_seq))
            else:
                self.train_data.append((gene_seq, exp_seq))

        train_samples = len(self.train_data)
        test_samples = len(self.test_data)
        logger.info(f"{train_samples:,} train_samples")
        logger.info(f"{test_samples:,} test_samples")
        return

    def create_train_batch(self, batch_size):
        """

        each sample in a batch: [gene id seq, gene expression seq, gene expression mask seq, hvg expression mask seq]
        """
        data = copy.deepcopy(self.train_data)
        random.shuffle(data)
        cells = len(data)
        batch_list = []

        for dl in range(0, cells, batch_size):
            dr = min(dl + batch_size, cells)
            batch = []

            for di in range(dl, dr):
                gene_seq, exp_seq = data[di]

                length = len(gene_seq)
                masks = round(length * mask_ratio)
                mask_seq = [0] * (length - masks) + [1] * masks
                random.shuffle(mask_seq)

                hvgmask_seq = [
                    1 if mask == 1 and (self.hvg_set is None or gene in self.hvg_set) else 0
                    for gene, mask in zip(gene_seq, mask_seq)
                ]

                batch.append([gene_seq, exp_seq, mask_seq, hvgmask_seq])
            batch_list.append(batch)

        return batch_list

    def create_test_batch(self, batch_size):
        """

        each sample in a batch: [gene id seq, gene expression seq, gene expression mask seq, hvg expression mask seq]
        """
        data = copy.deepcopy(self.test_data)
        cells = len(data)
        batch_list = []

        for dl in range(0, cells, batch_size):
            dr = min(dl + batch_size, cells)
            batch_list.append(data[dl:dr])

        return batch_list

    def create_test_no_mask_batch(self):
        """

        each sample in a batch: [gene id seq, gene expression seq, gene expression mask seq, hvg mask seq]
        """
        batch_list = []
        for test_index, (index_seq, exp_seq, mask_seq, hvgmask_seq) in enumerate(self.test_data):
            mask_seq = [0 for _ in mask_seq]
            batch_list.append([[index_seq, exp_seq, mask_seq, hvgmask_seq]])
        return batch_list


def forward_a_batch(batch, model, criterion, mask_input=True):
    exp_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor = GeneUtils.preprocess_autoencoder_batch(batch)

    exp_tensor = exp_tensor.to("cuda")
    maskedexp_tensor = maskedexp_tensor.to("cuda")
    mask_tensor = mask_tensor.to("cuda")
    hvgmask_tensor = hvgmask_tensor.to("cuda")

    if mask_input:
        o = model(maskedexp_tensor)
    else:
        o = model(exp_tensor)
    loss = criterion(o, exp_tensor)
    all_loss = loss.sum()
    alls = exp_tensor.shape[0] * exp_tensor.shape[1]

    mask_loss = (loss * mask_tensor).sum()
    masks = mask_tensor.sum()

    hvgmask_loss = (loss * hvgmask_tensor).sum()
    hvgmasks = hvgmask_tensor.sum()

    return o, all_loss, alls, mask_loss, masks, hvgmask_loss, hvgmasks


def evaluate_all_batch(batch_list, model, criterion):
    all_loss = 0
    alls = 0

    mask_loss = 0
    masks = 0

    hvgmask_loss = 0
    hvgmasks = 0

    model.eval()

    with torch.no_grad():
        for batch in batch_list:
            (
                _, batch_all_loss, batch_alls, batch_mask_loss, batch_masks, batch_hvgmask_loss, batch_hvgmasks,
            ) = forward_a_batch(batch, model, criterion, mask_input=True)

            all_loss += batch_all_loss.item()
            alls += batch_alls
            mask_loss += batch_mask_loss.item()
            masks += int(batch_masks.item())
            hvgmask_loss += batch_hvgmask_loss.item()
            hvgmasks += int(batch_hvgmasks.item())

    model.train()

    return all_loss, alls, mask_loss, masks, hvgmask_loss, hvgmasks


def run_train(arg):
    random.seed(42)
    torch.manual_seed(42)

    if arg.model.startswith("model_autoencoder__"):
        _, mask_input = arg.model.split("__")
        model = Model_autoencoder()
        mask_input = mask_input == "mask"
    else:
        assert False
    model.to("cuda")
    model.train()
    logger.info(f"Masking input when training: {mask_input}")

    lr = 1e-4 if "transformer" in arg.model else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = torch.nn.MSELoss(reduction="none")

    dataset = Dataset(arg.cell_gene_file, arg.cell_exp_file, arg.test_file, hvg_file=arg.hvg_file)

    train_batch_list = dataset.create_train_batch(arg.batch_size)
    train_batches = len(train_batch_list)
    logger.info(f"{train_batches:,} train batches")

    test_batch_list = dataset.create_test_batch(arg.batch_size)
    test_batches = len(test_batch_list)
    logger.info(f"{test_batches:,} test batches")

    train_batch_i = 0
    epoch = 1

    train_all_loss = 0
    train_alls = 0
    train_mask_loss = 0
    train_masks = 0
    train_hvgmask_loss = 0
    train_hvgmasks = 0

    best_test_result = (None, float("inf"), float("inf"), float("inf"))

    for train_step_i in range(1, 1 + arg.train_steps):
        log_prefix = f"({train_step_i:,}/{arg.train_steps:,})"

        # get training batch
        if train_batch_i == len(train_batch_list):
            train_batch_list = dataset.create_train_batch(arg.batch_size)
            train_batch_i = 0
            epoch += 1
        train_batch = train_batch_list[train_batch_i]
        train_batch_i += 1

        # train a batch
        optimizer.zero_grad()
        (
            o, batch_all_loss, batch_alls, batch_mask_loss, batch_masks, batch_hvgmask_loss, batch_hvgmasks,
        ) = forward_a_batch(train_batch, model, criterion, mask_input=mask_input)
        if mask_input:
            mean_loss = batch_mask_loss / batch_masks
        else:
            mean_loss = batch_all_loss / batch_alls
        mean_loss.backward()
        optimizer.step()

        # log loss
        train_all_loss += batch_all_loss.item()
        train_alls += batch_alls
        train_mask_loss += batch_mask_loss.item()
        train_masks += int(batch_masks.item())
        train_hvgmask_loss += batch_hvgmask_loss.item()
        train_hvgmasks += int(batch_hvgmasks.item())

        if train_step_i % arg.log_steps == 0:
            train_all_loss /= train_alls
            train_mask_loss /= train_masks
            train_hvgmask_loss /= train_hvgmasks
            logger.info(
                f"{log_prefix} [train]"
                f" all_loss={train_all_loss:.2e} alls={train_alls:,}"
                f" mask_loss={train_mask_loss:.2e} masks={train_masks:,}"
                f" hvgmask_loss={train_hvgmask_loss:.2e} hvgmasks={train_hvgmasks:,}"
            )
            train_all_loss = 0
            train_alls = 0
            train_mask_loss = 0
            train_masks = 0
            train_hvgmask_loss = 0
            train_hvgmasks = 0

        # evaluate on test
        if train_step_i % arg.test_steps == 0:
            (
                test_all_loss, test_alls, test_mask_loss, test_masks, test_hvgmask_loss, test_hvgmasks,
            ) = evaluate_all_batch(test_batch_list, model, criterion)
            test_all_loss /= test_alls
            test_mask_loss /= test_masks
            test_hvgmask_loss /= test_hvgmasks

            if test_mask_loss < best_test_result[2]:
                best_test_result = (train_step_i, test_all_loss, test_mask_loss, test_hvgmask_loss)
                best_log = " best"
                torch.save(model.state_dict(), arg.model_file)
            else:
                best_log = ""

            logger.info(
                f"{log_prefix} [test]"
                f" all_loss={test_all_loss:.2e} alls={test_alls:,}"
                f" mask_loss={test_mask_loss:.2e} masks={test_masks:,}"
                f" hvgmask_loss={test_hvgmask_loss:.2e} hvgmasks={test_hvgmasks:,}"
                f"{best_log}"
            )

    logger.info("Training complete")
    steps, test_all_loss, test_mask_loss, test_hvgmask_loss = best_test_result
    logger.info(
        f"Best test result:"
        f" steps={steps:,}"
        f" all_loss={test_all_loss:.2e}"
        f" mask_loss={test_mask_loss:.2e}"
        f" hvgmask_loss={test_hvgmask_loss:.2e}"
    )
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_gene_file", type=str, default="seq/gene_seq_index.csv")
    parser.add_argument("--cell_exp_file", type=str, default="seq/gene_seq_logexp.npy")
    parser.add_argument("--hvg_file", type=str, default="hvg/hvg_index.npy")

    parser.add_argument("--test_file", type=str, default="train/test.json")
    parser.add_argument("--model_file", type=str, default="train/model.pt")
    parser.add_argument("--weight_dir", type=str, default="train/weight")

    parser.add_argument("--source", type=str, default="pool")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=4336)

    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=200)
    parser.add_argument("--test_steps", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="model")
    arg = parser.parse_args()

    run_train(arg)
    # run_test(arg)
    # run_test_weight(arg)
    # tmp(arg)
    return


if __name__ == "__main__":
    main()
    sys.exit()
