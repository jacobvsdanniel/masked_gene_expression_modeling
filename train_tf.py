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

from model_tf import (
    GeneUtils,
    ModelTransformer,
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
    def __init__(self, cell_gene_file, cell_exp_file, test_file, hvg_file=None, tf_file=None):
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

        if tf_file:
            self.tf_set = set(read_npy(tf_file))
            tf_indices = len(self.tf_set)
            logger.info(f"{tf_indices:,} tf_indices")
        else:
            self.tf_set = None

        testcell_to_maskseq = dict(testcell_maskseq_data)

        for cell, (gene_seq, exp_seq) in enumerate(zip(cell_gene_data, cell_exp_data)):
            assert len(gene_seq) == len(exp_seq)
            gene_seq = [int(gene) for gene in gene_seq]
            exp_seq = exp_seq.tolist()
            tfmask_seq = [
                True if self.tf_set and gene not in self.tf_set else False
                for gene in gene_seq
            ]

            if cell in testcell_to_maskseq:
                mask_seq = testcell_to_maskseq[cell]
                assert len(gene_seq) == len(mask_seq)
                hvgmask_seq = [
                    1 if mask == 1 and (self.hvg_set is None or gene in self.hvg_set) else 0
                    for gene, mask in zip(gene_seq, mask_seq)
                ]
                self.test_data.append((gene_seq, exp_seq, tfmask_seq, mask_seq, hvgmask_seq))

            # else:
            #     self.train_data.append((gene_seq, exp_seq))
            self.train_data.append((gene_seq, exp_seq, tfmask_seq))

        train_samples = len(self.train_data)
        test_samples = len(self.test_data)
        logger.info(f"{train_samples:,} train_samples")
        logger.info(f"{test_samples:,} test_samples")
        return

    def create_train_batch(self, batch_size):
        """

        each sample in a batch: [
            gene id seq,
            gene expression seq,
            tf attention mask seq,
            gene expression mask seq,
            hvg expression mask seq,
        ]
        """
        data = copy.deepcopy(self.train_data)
        random.shuffle(data)
        cells = len(data)
        batch_list = []

        for dl in range(0, cells, batch_size):
            dr = min(dl + batch_size, cells)
            batch = []

            for di in range(dl, dr):
                gene_seq, exp_seq, tfmask_seq = data[di]

                length = len(gene_seq)
                masks = round(length * mask_ratio)
                mask_seq = [0] * (length - masks) + [1] * masks
                random.shuffle(mask_seq)

                hvgmask_seq = [
                    1 if mask == 1 and (self.hvg_set is None or gene in self.hvg_set) else 0
                    for gene, mask in zip(gene_seq, mask_seq)
                ]

                batch.append([gene_seq, exp_seq, tfmask_seq, mask_seq, hvgmask_seq])
            batch_list.append(batch)

        return batch_list

    def create_test_batch(self, batch_size):
        """

        each sample in a batch: [
            gene id seq,
            gene expression seq,
            tf attention mask seq,
            gene expression mask seq,
            hvg expression mask seq,
        ]
        """
        data = copy.deepcopy(self.test_data)
        cells = len(data)
        batch_list = []

        for dl in range(0, cells, batch_size):
            dr = min(dl + batch_size, cells)
            batch_list.append(data[dl:dr])

        return batch_list

    def create_weight_batch(self):
        """

        each batch contains only one sample
        batches ordered by data, no shuffling
        each sample in a batch: [
            gene id seq,
            gene expression seq,
            tf attention mask seq,
            gene expression mask seq,
            hvg expression mask seq,
        ]
        """
        data = copy.deepcopy(self.train_data)
        cells = len(data)
        batch_list = []

        for di in range(cells):
            gene_seq, exp_seq, tfmask_seq = data[di]

            length = len(gene_seq)
            masks = round(length * mask_ratio)
            mask_seq = [0] * (length - masks) + [1] * masks
            random.shuffle(mask_seq)

            hvgmask_seq = [
                1 if mask == 1 and (self.hvg_set is None or gene in self.hvg_set) else 0
                for gene, mask in zip(gene_seq, mask_seq)
            ]

            batch_list.append([[gene_seq, exp_seq, tfmask_seq, mask_seq, hvgmask_seq]])

        return batch_list


def forward_a_batch(batch, model, criterion):
    (
        gene_tensor, exp_tensor, tfmask_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor,
    ) = GeneUtils.preprocess_batch(batch)

    gene_tensor = gene_tensor.to("cuda")
    exp_tensor = exp_tensor.to("cuda")
    tfmask_tensor = tfmask_tensor.to("cuda")
    maskedexp_tensor = maskedexp_tensor.to("cuda")
    mask_tensor = mask_tensor.to("cuda")
    hvgmask_tensor = hvgmask_tensor.to("cuda")

    o = model(gene_tensor, maskedexp_tensor, tfmask_tensor)
    loss = criterion(o, exp_tensor)

    mask_loss = (loss * mask_tensor).sum()
    masks = mask_tensor.sum()

    hvgmask_loss = (loss * hvgmask_tensor).sum()
    hvgmasks = hvgmask_tensor.sum()

    return o, mask_loss, masks, hvgmask_loss, hvgmasks


def evaluate_all_batch(batch_list, model, criterion):
    loss = 0
    masks = 0

    hvg_loss = 0
    hvg_masks = 0

    model.eval()
    with torch.no_grad():
        for batch in batch_list:
            _, batch_loss, batch_masks, batch_hvg_loss, batch_hvg_masks = forward_a_batch(batch, model, criterion)
            loss += batch_loss.item()
            masks += int(batch_masks.item())
            hvg_loss += batch_hvg_loss.item()
            hvg_masks += int(batch_hvg_masks.item())
    model.train()

    loss /= masks
    hvg_loss /= hvg_masks
    return loss, masks, hvg_loss, hvg_masks


def run_train(arg):
    random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    if arg.model.startswith("model_transformer__"):
        _, layers, dimension, heads, dimension_feedforward = arg.model.split("__")
        layers = int(layers)
        dimension = int(dimension)
        heads = int(heads)
        dimension_feedforward = int(dimension_feedforward)
        model = ModelTransformer(layers, dimension, heads, dimension_feedforward)

    else:
        assert False
    model.to("cuda")
    model.train()

    lr = 1e-4 if "transformer" in arg.model else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    criterion = torch.nn.MSELoss(reduction="none")

    dataset = Dataset(
        arg.cell_gene_file, arg.cell_exp_file, arg.test_file,
        hvg_file=arg.hvg_file, tf_file=arg.tf_file,
    )

    train_batch_list = dataset.create_train_batch(arg.batch_size)
    train_batches = len(train_batch_list)
    logger.info(f"{train_batches:,} train batches")

    test_batch_list = dataset.create_test_batch(arg.batch_size)
    test_batches = len(test_batch_list)
    logger.info(f"{test_batches:,} test batches")

    train_batch_i = 0
    epoch = 1
    train_loss = 0
    train_masks = 0
    train_hvg_loss = 0
    train_hvg_masks = 0
    best_test_result = (None, float("inf"), float("inf"))

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
        _, batch_loss, batch_masks, batch_hvg_loss, batch_hvg_masks = forward_a_batch(train_batch, model, criterion)
        if arg.backward == "all":
            backward_loss = batch_loss
            backward_masks = batch_masks
        elif arg.backward == "hvg":
            backward_loss = batch_hvg_loss
            backward_masks = batch_hvg_masks
        else:
            assert False
        if int(backward_masks.item()) > 0:
            # should NOT write "backward_loss /= backward_masks"
            # otherwise as the tensor is modified in place, affecting batch_loss or batch_hvg_loss
            backward_loss = backward_loss / backward_masks
            backward_loss.backward()
            optimizer.step()

        # log loss
        train_loss += batch_loss.item()
        train_masks += int(batch_masks.item())
        train_hvg_loss += batch_hvg_loss.item()
        train_hvg_masks += int(batch_hvg_masks.item())
        if train_step_i % arg.log_steps == 0:
            train_loss /= train_masks
            train_hvg_loss /= train_hvg_masks
            logger.info(
                f"{log_prefix}"
                f" [train] loss={train_loss:.2e} masks={train_masks:,}"
                f" hvg_loss={train_hvg_loss:.2e} hvg_masks={train_hvg_masks:,}"
            )
            train_loss = 0
            train_masks = 0
            train_hvg_loss = 0
            train_hvg_masks = 0

        # evaluate on test
        if train_step_i % arg.test_steps == 0:
            test_loss, test_masks, test_hvg_loss, test_hvg_masks = evaluate_all_batch(test_batch_list, model, criterion)
            if arg.backward == "all" and test_loss < best_test_result[1]:
                best_test_result = (train_step_i, test_loss, test_hvg_loss)
                best_log = " best"
                torch.save(model.state_dict(), arg.model_file)
            elif arg.backward == "hvg" and test_hvg_loss < best_test_result[2]:
                best_test_result = (train_step_i, test_loss, test_hvg_loss)
                best_log = " best"
                torch.save(model.state_dict(), arg.model_file)
            else:
                best_log = ""
            logger.info(
                f"{log_prefix}"
                f" [test] loss={test_loss:.2e} masks={test_masks:,}"
                f" hvg_loss={test_hvg_loss:.2e} hvg_masks={test_hvg_masks:,}"
                f"{best_log}"
            )

    logger.info("Training complete")
    steps, test_loss, test_hvg_loss = best_test_result
    logger.info(f"Best test result: steps={steps:,} loss={test_loss:.2e} hvg_loss={test_hvg_loss:.2e}")
    return


def run_test(arg):
    if arg.model.startswith("model_transformer__"):
        _, layers, dimension, heads, dimension_feedforward = arg.model.split("__")
        layers = int(layers)
        dimension = int(dimension)
        heads = int(heads)
        dimension_feedforward = int(dimension_feedforward)
        model = ModelTransformer(layers, dimension, heads, dimension_feedforward)
    else:
        assert False
    model.to("cuda")
    model.load_state_dict(torch.load(arg.model_file))
    model.eval()

    criterion = torch.nn.MSELoss(reduction="none")
    dataset = Dataset(
        arg.cell_gene_file, arg.cell_exp_file, arg.test_file,
        hvg_file=arg.hvg_file, tf_file=arg.tf_file,
    )

    test_batch_list = dataset.create_test_batch(arg.batch_size)
    test_batches = len(test_batch_list)
    logger.info(f"{test_batches:,} test batches")

    loss, masks, hvg_loss, hvg_masks = evaluate_all_batch(test_batch_list, model, criterion)
    logger.info(
        f"[test] loss={loss:.2e} masks={masks:,}"
        f" hvg_loss={hvg_loss:.2e} hvg_masks={hvg_masks:,}"
    )
    return


def run_weight(arg):
    # model
    if arg.model.startswith("model_transformer__"):
        _, layers, dimension, heads, dimension_feedforward = arg.model.split("__")
        layers = int(layers)
        dimension = int(dimension)
        heads = int(heads)
        dimension_feedforward = int(dimension_feedforward)
        model = ModelTransformer(layers, dimension, heads, dimension_feedforward, attn_hook=True)
    else:
        assert False
    model.to("cuda")
    model.load_state_dict(torch.load(arg.model_file))
    model.eval()

    # dataset
    dataset = Dataset(
        arg.cell_gene_file, arg.cell_exp_file, arg.test_file,
        hvg_file=arg.hvg_file, tf_file=arg.tf_file,
    )
    batch_list = dataset.create_weight_batch()
    batches = len(batch_list)
    logger.info(f"{batches:,} batches")

    # gene-gene matrices
    os.makedirs(arg.weight_dir, exist_ok=False)
    with torch.no_grad():
        for bi, batch in enumerate(batch_list):
            bi += 1

            (
                gene_tensor, exp_tensor, tfmask_tensor, _maskedexp_tensor, _mask_tensor, _hvgmask_tensor,
            ) = GeneUtils.preprocess_batch(batch)
            gene_tensor = gene_tensor.to("cuda")
            exp_tensor = exp_tensor.to("cuda")
            tfmask_tensor = tfmask_tensor.to("cuda")

            w = model.get_attn_weight(gene_tensor, exp_tensor, tfmask_tensor, layer=arg.attn_layer).to("cpu")
            w = w[0]

            file = os.path.join(arg.weight_dir, f"{bi}.npy")
            np.save(file, w)

            if bi % 1000 == 0 or bi == batches:
                logger.info(f"{bi}/{batches}: {w.shape} saved")
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="tmp", choices=["train", "test", "weight", "tmp"])

    parser.add_argument("--cell_gene_file", type=str, default="seq/gene_seq_index.csv")
    parser.add_argument("--cell_exp_file", type=str, default="seq/gene_seq_logexp.npy")
    parser.add_argument("--hvg_file", type=str, default="hvg/hvg_index.npy")
    parser.add_argument("--tf_file", type=str, default="tf/tf_index.npy")

    parser.add_argument("--test_file", type=str, default="train/test.json")
    parser.add_argument("--model_file", type=str, default="train/model.pt")
    parser.add_argument("--weight_dir", type=str, default="train/weight")

    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=200)
    parser.add_argument("--test_steps", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument("--backward", type=str, default="all", choices=["all", "hvg"])
    parser.add_argument("--attn_layer", type=str, default="last", choices=["last", "all", "first"])
    parser.add_argument("--seed", type=int, default=42)

    arg = parser.parse_args()
    for key, value in vars(arg).items():
        logger.info(f"[{key}] {value}")

    if arg.task == "train":
        run_train(arg)

    elif arg.task == "test":
        run_test(arg)

    elif arg.task == "weight":
        run_weight(arg)

    elif arg.task == "tmp":
        pass

    return


if __name__ == "__main__":
    main()
    sys.exit()
