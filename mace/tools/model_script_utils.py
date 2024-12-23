import ast
import logging

import numpy as np
from e3nn import o3

from mace import modules
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model


def configure_model(
    args, train_loader, atomic_energies, model_foundation=None, heads=None, z_table=None
):

    output_args = {
        "energy": args.compute_energy
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        logging.info("No scaling selected. args.std = 1.0 ")
    elif (args.mean is None or args.std is None):
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )

    # Build model
    #if model_foundation is not None and args.model in ["MACE", "ScaleShiftMACE"]:
    #else:
    logging.info("Building model")
    logging.info(
        f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
    )
    logging.info(
        f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
    )
    logging.info(
        f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
    )
    logging.info(
        f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
    )
    logging.info(
        f"Distance transform for radial basis functions: {args.distance_transform}"
    )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    logging.info(f"Hidden irreps: {args.hidden_irreps}")

    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )
    model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    return model, output_args

def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements
    if args.model == "MACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads), ##
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "ScaleShiftMACE": ##MACE+. custom interaction block. custom radial basis
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion, ###
            distance_transform=args.distance_transform, ###
            correlation=args.correlation, ###
            gate=modules.gate_dict[args.gate], #activation function
            interaction_cls_first=modules.interaction_classes[args.interaction_first], ##
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std, 
            atomic_inter_shift=args.mean, ## 
            radial_MLP=ast.literal_eval(args.radial_MLP), ###
            radial_type=args.radial_type, ###
            heads=heads, ###
        )

    if args.model == "ScaleShiftBOTNet": ##
        return modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
        )
    if args.model == "BOTNet":
        return modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    
   
    raise RuntimeError(f"Unknown model: '{args.model}'")
