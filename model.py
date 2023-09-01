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
        length_list = [len(gene_seq) for gene_seq, exp_seq, mask_seq, hvgmask_seq in batch]
        max_length = max(length_list)

        gene_tensor = []
        exp_tensor = []
        maskedexp_tensor = []
        mask_tensor = []
        hvgmask_tensor = []

        for bi, (gene_seq, exp_seq, mask_seq, hvgmask_seq) in enumerate(batch):
            length = length_list[bi]
            assert length == len(gene_seq) == len(exp_seq) == len(mask_seq) == len(hvgmask_seq)

            gene_tensor.append(
                [gene + 1 for gene in gene_seq] + [cls.pad_id] * (max_length - length)  # gene id 0 is for pad_id
            )
            exp_tensor.append(
                exp_seq + [cls.exp_mask_value] * (max_length - length)
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
        maskedexp_tensor = torch.tensor(maskedexp_tensor, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
        hvgmask_tensor = torch.tensor(hvgmask_tensor, dtype=torch.float32)

        return gene_tensor, exp_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor, length_list

    @classmethod
    def preprocess_autoencoder_batch(cls, batch):
        exp_tensor = []
        maskedexp_tensor = []
        mask_tensor = []
        hvgmask_tensor = []

        for bi, (gene_seq, exp_seq, mask_seq, hvgmask_seq) in enumerate(batch):
            assert len(gene_seq) == len(exp_seq) == len(mask_seq) == len(hvgmask_seq)

            exp_array = [cls.exp_mask_value] * cls.genes
            maskedexp_array = [cls.exp_mask_value] * cls.genes
            mask_arrary = [0] * cls.genes
            hvgmask_arrary = [0] * cls.genes

            for gene, exp, mask, hvgmask in zip(gene_seq, exp_seq, mask_seq, hvgmask_seq):
                exp_array[gene] = exp
                maskedexp_array[gene] = exp if mask == 0 else cls.exp_mask_value
                mask_arrary[gene] = mask
                hvgmask_arrary[gene] = hvgmask

            exp_tensor.append(exp_array)
            maskedexp_tensor.append(maskedexp_array)
            mask_tensor.append(mask_arrary)
            hvgmask_tensor.append(hvgmask_arrary)

        exp_tensor = torch.tensor(exp_tensor, dtype=torch.float32)
        maskedexp_tensor = torch.tensor(maskedexp_tensor, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)
        hvgmask_tensor = torch.tensor(hvgmask_tensor, dtype=torch.float32)

        return exp_tensor, maskedexp_tensor, mask_tensor, hvgmask_tensor


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
        :param m: [B, L]
        :return: [B, L, D]
        """
        return self.transformer(x, src_key_padding_mask=m)


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


class ModelAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer_list = torch.nn.ModuleList([
            torch.nn.Linear(in_features=GeneUtils.genes, out_features=512, bias=True),
            torch.nn.Linear(in_features=512, out_features=128, bias=True),
        ])
        self.decoder_layer_list = torch.nn.ModuleList([
            torch.nn.Linear(in_features=128, out_features=512, bias=True),
            torch.nn.Linear(in_features=512, out_features=GeneUtils.genes, bias=True),
        ])
        return

    def forward(self, ge):
        """

        :param ge: [B, G], masked gene expression
        :return: [B, G], predicted gene expression
        """
        for layer in self.encoder_layer_list:
            ge = layer(ge)
            ge = functional.relu(ge)
        for layer in self.decoder_layer_list[:-1]:
            ge = layer(ge)
            ge = torch.sigmoid(ge)
        ge = self.decoder_layer_list[-1](ge)
        return ge


class ModelTransformer(torch.nn.Module):
    def __init__(self, layers, dimension, heads, dimension_feedforward, attn_hook=False):
        super().__init__()
        self.embedding = Embedding(dimension - 1)
        self.transformer = Transformer(layers, dimension, heads, dimension_feedforward)
        self.classifier = Classifier(dimension)
        self.layers = layers

        if attn_hook:
            self.module_to_io = {}

            def hook(module, i, o):
                self.module_to_io[module] = (i, o)
                return

            for layer in self.transformer.transformer.layers:
                layer.self_attn.register_forward_hook(hook=hook)
        return

    def forward(self, gi, ge, ll):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param ll: [B], genes
        :return: [B, L], predicted gene expression
        """
        x1 = self.embedding(gi)
        # x1: [B, L, D-1]

        x2 = ge.unsqueeze(2)
        # x2: [B, L, 1]

        x = torch.cat([x1, x2], dim=2)
        # x: [B, L, D]

        m = get_mask(ll, False, True, torch.bool)
        # m: [B, L]

        h = self.transformer(x, m)
        # h: [B, L, D]

        o = self.classifier(h)
        # o: [B, L]
        return o

    def get_attn_weight(self, gi, ge, ll, layer="last"):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param ll: [B], genes
        :param layer: last / all / first
        :return: [B, L], predicted gene expression
        """
        x1 = self.embedding(gi)
        # x1: [B, L, D - 1]

        x2 = ge.unsqueeze(2)
        # x2: [B, L, 1]

        x = torch.cat([x1, x2], dim=2)
        # x: [B, L, D]

        m = get_mask(ll, False, True, torch.bool)
        # m: [B, L]

        _ = self.transformer(x, m)
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
                key_padding_mask=m, need_weights=True, average_attn_weights=True,
            )
            # o: [B, L, D]
            # w: [B, L, L], row-col: tgt-src (each row sum to 1)
            assert (attn_o[0] == o).all()
            w_sum = w if w_sum is None else w_sum + w

        w_mean = w_sum / len(layer_list)
        return w_mean


class Model_feedforward_only_self_geneid(torch.nn.Module):
    def __init__(self, dimension, layers):
        super().__init__()
        self.embedding = Embedding(dimension)
        self.feedforward = FeedForward(dimension, layers)
        self.classifier = Classifier(dimension)
        return

    def forward(self, gi, ge, l):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param l: [B], genes
        :return: [B, L], predicted gene expression
        """
        x = self.embedding(gi)
        # x1: [B, L, D]

        h = self.feedforward(x)
        # h: [B, L, D]

        o = self.classifier(h)
        # o: [B, L]
        return o


class Model_feedforward_none(torch.nn.Module):
    def __init__(self, dimension, layers):
        super().__init__()
        self.dimension = dimension
        self.feedforward = FeedForward(dimension, layers)
        self.classifier = Classifier(dimension)
        return

    def forward(self, gi, ge, l):
        """

        :param gi: [B, L], gene index
        :param ge: [B, L], gene expression
        :param l: [B], genes
        :return: [B, L], predicted gene expression
        """
        B, L = gi.shape
        x = torch.zeros((B, L, self.dimension), dtype=torch.float32).to("cuda")
        # x: [B, L, D]

        h = self.feedforward(x)
        # h: [B, L, D]

        o = self.classifier(h)
        # o: [B, L]
        return o


def main():
    torch.manual_seed(42)

    model_parameter_list = [
        (ModelTransformer, 4, 512, 8, 2048),
        (Model_feedforward_only_self_geneid, 100, 6),
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
