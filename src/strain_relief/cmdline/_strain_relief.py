import hydra
from omegaconf import DictConfig

from strain_relief import compute_strain
from strain_relief.io import load_parquet


@hydra.main(version_base=None, config_path="../hydra_config", config_name="default")
def main(cfg: DictConfig):
    df = load_parquet(**cfg.io.input)
    return compute_strain(df=df, cfg=cfg)


if __name__ == "__main__":
    torsions = main()
