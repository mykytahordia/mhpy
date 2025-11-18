defaults:
  - model: default
  - train: default
  - data: default
  - _self_

# Hydra-specific configuration
hydra:
  run:
    dir: logs/hydra/${hydra.job.name}/singlerun/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: logs/hydra/${hydra.job.name}/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  job:
    chdir: False