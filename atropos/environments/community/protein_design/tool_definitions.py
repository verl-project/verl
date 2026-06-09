PREDICT_TARGET_STRUCTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_target_structure_alphafold2",
        "description": "Predicts the 3D structure of the target protein sequence using AlphaFold2.",
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence of the target protein.",
                },
            },
            "required": ["sequence"],
        },
    },
}

DESIGN_BINDER_BACKBONE_TOOL = {
    "type": "function",
    "function": {
        "name": "design_binder_backbone_rfdiffusion",
        "description": (
            "Generates a novel protein binder backbone using RFDiffusion, "
            "conditioned on the target protein structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contigs": {
                    "type": "string",
                    "description": "RFDiffusion contigs (e.g., 'A1-100/0 50-70').",
                },
                "hotspot_residues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional hotspot residues (e.g., ['A50', 'A52']).",
                },
            },
            "required": ["contigs"],
        },
    },
}

DESIGN_BINDER_SEQUENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "design_binder_sequence_proteinmpnn",
        "description": "Designs an amino acid sequence for the generated binder backbone.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_temp": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Sampling temperatures (e.g., [0.1, 0.2]). Default [0.1]."
                    ),
                }
            },
            "required": [],
        },
    },
}

EVALUATE_COMPLEX_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_binder_complex_alphafold2_multimer",
        "description": (
            "Predicts the complex structure of target and designed binder, "
            "providing quality scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relax_prediction": {
                    "type": "boolean",
                    "description": "Whether to relax the prediction. Default True.",
                }
            },
            "required": [],
        },
    },
}

ALL_TOOLS_LIST = [
    PREDICT_TARGET_STRUCTURE_TOOL,
    DESIGN_BINDER_BACKBONE_TOOL,
    DESIGN_BINDER_SEQUENCE_TOOL,
    EVALUATE_COMPLEX_TOOL,
]
