import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a specialized AI system for de novo protein design via a staged simulation loop. "
    "Your objective is to generate binder sequences that are structurally and functionally "
    "optimized to bind a given target protein.\n\n"
    "You will be guided through a multi-step pipeline:\n\n"
    "1. Structure prediction (AlphaFold)\n"
    "2. Binder backbone generation (RFdiffusion)\n"
    "3. Sequence design (ProteinMPNN)\n"
    "4. Structure evaluation (AlphaFold-Multimer)\n"
    "5. Feedback loop\n\n"
    "You must always:\n"
    "- Respect the required file format for each tool (e.g., FASTA, PDB).\n"
    "- Structure your outputs cleanly so they can be parsed and executed programmatically.\n"
    "- Be explicit in all configuration steps (e.g., contigs, hotspots).\n"
    "- Minimize ambiguity or verbosity; prefer concise and functional outputs.\n"
    "- Reason step-by-step when appropriate."
)


def construct_user_prompt(state: dict) -> str:
    """
    Constructs the appropriate user prompt for the current internal step.

    Args:
        state: The current workflow state dictionary (from episodes_state)

    Returns:
        A formatted user prompt string for the current step
    """
    internal_step = state.get("current_internal_step", 0)
    target_sequence = state.get("target_sequence")
    user_prompt_str = ""

    if internal_step == 0:
        user_prompt_str = (
            f"The target protein sequence is: {target_sequence}. "
            "Your first task is to predict its 3D structure using the 'predict_target_structure_alphafold2' tool. "
            "You must provide the 'sequence' argument."
        )
    elif internal_step == 1:

        chain_details = state.get("target_chain_details", {})
        if chain_details:
            chain_info_parts = []
            for chain_id, details in chain_details.items():
                min_r = details.get("min_residue", "N/A")
                max_r = details.get("max_residue", "N/A")
                length = details.get("length", "N/A")
                chain_info_parts.append(
                    f"Chain {chain_id} (Residues: {min_r}-{max_r}, Length: {length} amino acids)"
                )
            chain_info_str = "\n- ".join(chain_info_parts)
            if chain_info_str:
                chain_info_str = "- " + chain_info_str
        else:
            chain_info_str = "Chain information not available or PDB not yet processed."

        user_prompt_str = (
            f"The 3D structure of the target protein has been predicted.\n"
            f"Target Protein Chain Details:\n{chain_info_str}\n\n"
            "Your task is to design a binder backbone using the 'design_binder_backbone_rfdiffusion' tool. "
            "You MUST specify 'contigs' for this tool. The 'contigs' string defines segments from the target PDB "
            "and segments for the new binder. "
            "Examples:\n"
            "  - To use residues 10 through 100 of target chain A, and then diffuse a 60-residue binder: "
            "'A10-100/0 60'\n"
            "  - To use chain B from residue 5 to 50, then diffuse a 30-residue binder, then use chain B "
            "from residue 60 to 100: 'B5-50/0 30 B60-100'\n"
            "You MUST use the chain IDs and residue ranges exactly as provided in the "
            "'Target Protein Chain Details' above. "
            "Do not invent chains or residue numbers outside these specified ranges for the target segments. "
            "For binder segments (e.g., '/0 60'), specify the desired length (e.g., 60).\n"
            "Optionally, provide 'hotspot_residues' (e.g., ['A50', 'A52']), ensuring these residues exist "
            "on the target as per the details above."
        )
    elif internal_step == 2:
        binder_pdb_content = state.get("binder_backbone_pdb_content")

        binder_pdb_preview = state.get(
            "binder_pdb_preview", "Binder PDB preview not available."
        )
        binder_chain_info_str = "Binder chain information not available."

        if binder_pdb_content:
            binder_chain_details = state.get("binder_chain_details", {})

            if binder_chain_details:
                chain_info_parts = []
                for cID, d_details in binder_chain_details.items():
                    min_r = d_details.get("min_residue", "N/A")
                    max_r = d_details.get("max_residue", "N/A")
                    length = d_details.get("length", "N/A")
                    chain_info_parts.append(
                        f"Chain {cID} (Residues: {min_r}-{max_r}, Length: {length} amino acids)"
                    )
                binder_chain_info_str = "\n- ".join(chain_info_parts)
                if binder_chain_info_str:
                    binder_chain_info_str = "- " + binder_chain_info_str
            else:
                binder_chain_info_str = "Binder chain details not found in state (expected from RFDiffusion)."

        else:
            pass

        user_prompt_str = (
            f"A binder backbone has been generated. Binder PDB preview:\n{binder_pdb_preview}\n"
            f"Binder chain information:\n{binder_chain_info_str}.\n"
            "Now, design an optimal amino acid sequence for this binder backbone using the "
            "'design_binder_sequence_proteinmpnn' tool. "
            "You can optionally specify 'sampling_temp' (e.g., [0.1, 0.2])."
        )
    elif internal_step == 3:
        designed_binder_seq_data = state.get("designed_binder_sequence")

        binder_display_str = "Not available"
        if isinstance(designed_binder_seq_data, list) and designed_binder_seq_data:
            if len(designed_binder_seq_data) == 1:
                binder_display_str = designed_binder_seq_data[0]
            else:
                binder_display_str = (
                    f"{len(designed_binder_seq_data)} chains: "
                    + ", ".join(
                        [
                            f"Chain {i+1} ({len(s)} aa): {s[:20]}..."
                            for i, s in enumerate(designed_binder_seq_data)
                        ]
                    )
                )
        elif isinstance(designed_binder_seq_data, str):
            binder_display_str = designed_binder_seq_data

        user_prompt_str = (
            f"A binder has been designed. Designed binder sequence(s): {binder_display_str}. "
            f"The original target sequence was: {target_sequence[:60]}...\n"
            "Finally, evaluate the binding complex of the original target protein and ALL chains of this "
            "designed binder using the 'evaluate_binder_complex_alphafold2_multimer' tool. "
            "You can optionally specify 'relax_prediction' (default is True)."
        )
    else:
        user_prompt_str = (
            "The protein design workflow is complete. No further actions required by you for this item. "
            "If successful, the key metric was the pLDDT of the complex."
        )

    if state.get("retry_count_this_internal_step", 0) > 0 and internal_step < 4:
        retry_prefix = "Your previous attempt at this step was not successful. "
        if state.get("previous_tool_error_message"):
            retry_prefix += f"Details: {state['previous_tool_error_message']}. "
        retry_prefix += (
            "Please review the requirements and PDB details carefully and try again to correctly use "
            "the expected tool.\n\n"
        )
        user_prompt_str = retry_prefix + user_prompt_str

    return user_prompt_str
