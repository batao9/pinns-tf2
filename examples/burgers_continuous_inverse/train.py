from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import rootutils
import tensorflow as tf
from omegaconf import DictConfig

import pinnstf


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstf.utils.load_data(root_path, "burgers_shock.mat")
    exact_u = np.real(data["usol"])

    return {"u": exact_u}


def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           t: tf.Tensor,
           extra_variables: Dict[str, tf.Tensor]): 
    """Define the partial differential equations (PDEs)."""

    u_x, u_t = pinnstf.utils.gradient(outputs["u"], [x, t])
    u_xx = pinnstf.utils.gradient(u_x, x)

    outputs["f"] = u_t + extra_variables["l1"] * outputs["u"] * u_x - tf.exp(extra_variables["l2"]) * u_xx

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstf.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstf.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
