import sys
import logging

import torch
import torch.nn.functional as functional

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def get_mask(length_list, pass_value, mask_value, mask_type):
    max_length = max(length_list)
    mask = [
        [pass_value] * length + [mask_value] * (max_length - length)
        for length in length_list
    ]
    mask = torch.tensor(mask, dtype=mask_type).to("cuda")
    # mask: [B, L]
    return mask


class GeneUtils:
    pad_id = 0
    exp_mask_value = -10

    genes = 16656
    tokens = 1 + genes

    @classmethod
    def preprocess_batch(cls, batch):
        length_list = [len(gene_seq) for gene_seq, exp_seq, tfmask_seq, mask_seq, hvgmask_seq in batch]
        max_length = max(length_list)

        gene_tensor = []
        exp_tensor = []
        maskedexp_tensor = []
        tfmask_tensor = []
        mask_tensor = []
        hvgmask_tensor = []

        for bi, (gene_seq, exp_seq, tfmask_seq, mask_seq, hvgmask_seq) in enumerate(batch):
            length = length_list[bi]
            assert length == len(gene_seq) == len(exp_seq) == len(mask_seq) == len(hvgmask_seq)

            gene_tensor.append(
                [gene + 1 for gene in gene_seq] + [cls.pad_id] * (max_length - length)  # gene id 0 is for pad_id
            )
            exp_tensor.append(
                exp_seq + [cls.exp_mask_value] * (max_length - length)
            )
            tfmask_tensor.append(
                tfmask_seq + [float("-inf")] * (max_length - length)
            )
            maskedexp_tensor.append(
                [exp if mask == 0 else cls.exp_mask_value for exp, mask in zip(exp_seq, mask_seq)]
                + [cls.exp_mask_value] * (max_length - length)
            )
            mask_tensor.append(
                mask_seq + [0] * (max_length - length)
            )
            hvgmask_tensor.append(
                hvgmask_seq + [0] * (max_length - length)
            )

        gene_tensor = torch.tensor(gene_tensor, dtype=torch.int32)
        exp_tensor = torch.tensor(exp_tensor, dtype=torch.float32)
        tfmask_tensor = torch.tensor(tfmask_tensor, dtype=torch.float32)
        maskedexp_tensor = torch.tensor(maskedexp_tensor, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
        hvgmask_tensor = torch.tensor(hvgmask_tensor, dtype=torch.float32)

        return gene_tensor, exp_tensor, tfmask_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor


class Embedding(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.embedding = torch.nn.Embedding(GeneUtils.tokens, dimension, padding_idx=GeneUtils.pad_id)
        return

    def forward(self, x):
        return self.embedding(x)


class Transformer(torch.nn.Module):
    def __init__(self, layers, dimension, heads, dimension_feedforward):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=layers,
            norm=None,
        )
        return

    def forward(self, x, m):
        """

        :param x: [B, L, D]
        :param m: [B * H, L, L]
        :return: [B, L, D]
        """
        return self.transformer(x, mask=m)


class FeedForward(torch.nn.Module):
    def __init__(self, dimension, layers):
        super().__init__()
        self.layer_list = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=dimension,
                out_features=dimension,
                bias=True,
            )
            for _ in range(layers)
        ])
        return

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
            x = functional.relu(x)
        return x


class Classifier(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=dimension,
            out_features=1,
            bias=True,
        )
        return

    def forward(self, x):
        o = self.linear(x)
        o = o.squeeze(dim=2)
        return o


class ModelTransformer(torch.nn.Module):
    def __init__(
            self, layers, dimension, heads, dimension_feedforward,
            initial_tf_weight, train_tf_weight,
            attn_hook=False,
    ):
        super().__init__()
        self.embedding = Embedding(dimension - 1)
        self.transformer = Transformer(layers, dimension, heads, dimension_feedforward)
        self.classifier = Classifier(dimension)
        self.layers = layers
        self.heads = heads

        assert initial_tf_weight > 0
        tf_weight = torch.tensor(initial_tf_weight, dtype=torch.float32)
        tf_weight = tf_weight.log()

        if train_tf_weight:
            self.tf_weight = torch.nn.parameter.Parameter(tf_weight)
        else:
            self.tf_weight = tf_weight

        if attn_hook:
            self.module_to_io = {}

            def hook(module, i, o):
                self.module_to_io[module] = (i, o)
                return

            for layer in self.transformer.transformer.layers:
                layer.self_attn.register_forward_hook(hook=hook)
        return

    def get_tf_weight(self):
        return self.tf_weight.exp().item()

    def forward(self, gi, ge, am):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param am: [B, L], attention mask
            0 -> normal gene
            1 -> tf
            -inf -> padding
        :return: [B, L], predicted gene expression
        """
        x1 = self.embedding(gi)
        # x1: [B, L, D-1]

        x2 = ge.unsqueeze(2)
        # x2: [B, L, 1]

        x = torch.cat([x1, x2], dim=2)
        # x: [B, L, D]

        H = self.heads
        B, L = am.shape
        am = am.unsqueeze(1).expand((B, H * L, L)).reshape((B * H, L, L))
        # am: [B * H, L, L]

        am = self.tf_weight.exp() * am
        # am: [B * H, L, L]

        h = self.transformer(x, am)
        # h: [B, L, D]

        o = self.classifier(h)
        # o: [B, L]
        return o

    def get_attn_weight(self, gi, ge, am, layer="last"):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param am: [B, L], attention mask
            0 -> normal gene
            1 -> tf
            -inf -> padding
        :param layer: last / all / first
        :return: [B, L], predicted gene expression
        """
        x1 = self.embedding(gi)
        # x1: [B, L, D - 1]

        x2 = ge.unsqueeze(2)
        # x2: [B, L, 1]

        x = torch.cat([x1, x2], dim=2)
        # x: [B, L, D]

        H = self.heads
        B, L = am.shape
        am = am.unsqueeze(1).expand((B, H * L, L)).reshape((B * H, L, L))
        # am: [B * H, L, L]

        am = self.tf_weight.exp() * am
        # am: [B * H, L, L]

        _ = self.transformer(x, am)
        # h: [B, L, D]

        if layer == "last":
            layer_list = [-1]
        elif layer == "all":
            layer_list = [i for i in range(self.layers)]
        elif layer == "first":
            layer_list = [0]
        else:
            assert False

        w_sum = None
        for li in layer_list:
            attn = self.transformer.transformer.layers[li].self_attn
            attn_i, attn_o = self.module_to_io[attn]
            o, w = attn(
                attn_i[0], attn_i[1], attn_i[2],
                attn_mask=am, need_weights=True, average_attn_weights=True,
            )
            # o: [B, L, D]
            # w: [B, L, L], row-col: tgt-src (each row sum to 1)
            assert (attn_o[0] == o).all()
            w_sum = w if w_sum is None else w_sum + w

        w_mean = w_sum / len(layer_list)
        return w_mean


def main():
    torch.manual_seed(42)

    model_parameter_list = [
        (ModelTransformer, 4, 512, 8, 2048),
    ]

    model_list = []
    for model_parameter in model_parameter_list:
        model_class, parameter_tuple = model_parameter[0], model_parameter[1:]
        model = model_class(*parameter_tuple)
        model.to("cuda")
        model_list.append(model)

    batch = [
        ([0, 1, 2], [1.5, 2.5, 3.5], [0, 1, 0], [0, 1, 0]),
        ([3, 4], [7.5, 3.5], [1, 0], [1, 0]),
    ]

    (
        gene_tensor, exp_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor, length_list,
    ) = GeneUtils.preprocess_batch(batch)
    gene_tensor = gene_tensor.to("cuda")
    maskedexp_tensor = maskedexp_tensor.to("cuda")

    for model in model_list:
        logger.info(f"{model}")
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{trainable_parameters:,} trainable_parameters")

        model.eval()
        with torch.no_grad():
            o = model(gene_tensor, maskedexp_tensor, length_list)
        o = o.cpu().numpy()

        logger.info(f"batch: {batch}")
        logger.info(f"gi: {gene_tensor}")
        logger.info(f"ge: {maskedexp_tensor}")
        logger.info(f"gm: {mask_tensor}")
        logger.info(f"l: {length_list}")
        logger.info(f"output: {o}")
    return


if __name__ == "__main__":
    main()
    sys.exit()
