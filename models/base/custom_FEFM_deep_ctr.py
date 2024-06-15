# -*- coding:utf-8 -*-
"""
Author:
    Harshit Pande

Reference:
    [1] Field-Embedded Factorization Machines for Click-through Rate Prediction]
    (https://arxiv.org/pdf/2009.09931.pdf)

    this file also supports all the possible Ablation studies for reproducibility

"""

from itertools import chain

from deepctr.feature_column import (
    DEFAULT_GROUP_NAME,
    build_input_features,
    get_linear_logit,
    input_from_feature_columns,
)
from deepctr.layers.core import DNN
from deepctr.layers.interaction import FEFMLayer
from deepctr.layers.utils import add_func, combined_dnn_input, concat_func, reduce_sum
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow.python.keras.models import Model

from models.base.core import PredictionLayer


def DeepFEFM(
    linear_feature_columns,
    dnn_feature_columns,
    use_fefm=True,
    dnn_hidden_units=(256, 128, 64),
    l2_reg_linear=0.00001,
    l2_reg_embedding_feat=0.00001,
    l2_reg_embedding_field=0.00001,
    l2_reg_dnn=0,
    seed=1024,
    dnn_dropout=0.0,
    exclude_feature_embed_in_dnn=False,
    use_linear=True,
    use_fefm_embed_in_dnn=True,
    dnn_activation="relu",
    dnn_use_bn=False,
    task="binary",
    num_classes=2,
):
    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(
        features,
        linear_feature_columns,
        l2_reg=l2_reg_linear,
        seed=seed,
        prefix="linear",
    )

    group_embedding_dict, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding_feat, seed, support_group=True
    )

    fefm_interaction_embedding = concat_func(
        [
            FEFMLayer(regularizer=l2_reg_embedding_field)(concat_func(v, axis=1))
            for k, v in group_embedding_dict.items()
            if k in [DEFAULT_GROUP_NAME]
        ],
        axis=1,
    )

    dnn_input = combined_dnn_input(
        list(chain.from_iterable(group_embedding_dict.values())), dense_value_list
    )

    if use_fefm_embed_in_dnn:
        if exclude_feature_embed_in_dnn:
            # Ablation3: remove feature vector embeddings from the DNN input
            dnn_input = fefm_interaction_embedding
        else:
            # No ablation
            dnn_input = concat_func([dnn_input, fefm_interaction_embedding], axis=1)

    dnn_out = DNN(
        dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed
    )(dnn_input)

    dnn_logit = Dense(
        1,
        use_bias=False,
    )(dnn_out)

    fefm_logit = Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(
        fefm_interaction_embedding
    )

    if (
        len(dnn_hidden_units) == 0 and use_fefm is False and use_linear is True
    ):  # only linear
        final_logit = linear_logit
    elif (
        len(dnn_hidden_units) == 0 and use_fefm is True and use_linear is True
    ):  # linear + FEFM
        final_logit = add_func([linear_logit, fefm_logit])
    elif (
        len(dnn_hidden_units) > 0 and use_fefm is False and use_linear is True
    ):  # linear +　Deep # Ablation1
        final_logit = add_func([linear_logit, dnn_logit])
    elif (
        len(dnn_hidden_units) > 0 and use_fefm is True and use_linear is True
    ):  # linear + FEFM + Deep
        final_logit = add_func([linear_logit, fefm_logit, dnn_logit])
    elif (
        len(dnn_hidden_units) == 0 and use_fefm is True and use_linear is False
    ):  # only FEFM (shallow)
        final_logit = fefm_logit
    elif (
        len(dnn_hidden_units) > 0 and use_fefm is False and use_linear is False
    ):  # only Deep
        final_logit = dnn_logit
    elif (
        len(dnn_hidden_units) > 0 and use_fefm is True and use_linear is False
    ):  # FEFM + Deep # Ablation2
        final_logit = add_func([fefm_logit, dnn_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(task, num_classes=num_classes)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
