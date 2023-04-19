from src.sampling.ld import LangevinDiffusionDAIS, LangevinDiffusionDSMCS
from src.sampling.ld_momentum import (
    LangevinMomentumDiffusionDAIS,
    LangevinMomentumDiffusionDSMCS,
)

ais_models = {
    # Langevin
    "ULA": LangevinDiffusionDAIS,
    # Langevin + Momentum
    "UHA": LangevinMomentumDiffusionDAIS,
}

smcs_models = {
    # Langevin
    "ULA": LangevinDiffusionDSMCS,
    # Langevin + Momentum
    "UHA": LangevinMomentumDiffusionDSMCS,
}


def get_ais_model(args, log_joint, log_variational, deltas, betas):
    # DAIS model (no resampling and DAIS bound, i.e. NO DSMCS bound)
    if args.resampling is None and not args.dsmcs:
        ais = ais_models[args.variational_augmentation](
            args, log_joint, log_variational, deltas, betas
        )
    # DSMCS model (resampling and DSMCS bound)
    elif args.resampling is not None and args.dsmcs:
        ais = smcs_models[args.variational_augmentation](
            args, log_joint, log_variational, deltas, betas
        )
    else:
        raise NotImplementedError
    return ais
