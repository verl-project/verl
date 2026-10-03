import logging
from typing import Dict, Set, Tuple, Union

logger = logging.getLogger(__name__)


def get_pdb_chain_details(
    pdb_content: str, preview_lines: int = 10
) -> Tuple[Dict[str, Dict[str, int]], str]:
    """
    Parses PDB content to extract detailed information for each chain.

    Returns:
        A tuple containing:
        - chain_details (Dict[str, Dict[str, int]]):
            A dictionary where keys are chain IDs (e.g., "A").
            Each value is another dictionary:
            {
                "min_residue": int,  # Smallest residue number found for this chain
                "max_residue": int,  # Largest residue number found for this chain
                "length": int       # Count of unique C-alpha atoms (residues) in this chain
            }
        - pdb_preview (str): A string preview of the PDB content.
    """
    chain_info_temp: Dict[str, Dict[str, Union[Set[int], int]]] = {}
    atom_lines = []
    header_lines = []

    for line in pdb_content.splitlines():
        if line.startswith("ATOM"):
            atom_lines.append(line)
            chain_id = line[21:22].strip()
            if not chain_id:
                chain_id = " "
            atom_name = line[12:16].strip()
            try:
                residue_num = int(line[22:26].strip())
                if chain_id not in chain_info_temp:
                    chain_info_temp[chain_id] = {"residues": set(), "ca_count": 0}
                chain_info_temp[chain_id]["residues"].add(residue_num)
                if atom_name == "CA":
                    chain_info_temp[chain_id]["ca_count"] += 1
            except ValueError:
                logger.warning(f"Could not parse residue number from PDB line: {line}")
                continue
        elif (
            line.startswith("HEADER")
            or line.startswith("TITLE")
            or line.startswith("COMPND")
        ):
            header_lines.append(line)

    chain_details: Dict[str, Dict[str, int]] = {}
    for chain_id, data in chain_info_temp.items():
        if data["residues"]:
            min_res = min(data["residues"])
            max_res = max(data["residues"])
            length = data["ca_count"] if data["ca_count"] > 0 else len(data["residues"])
            chain_details[chain_id] = {
                "min_residue": min_res,
                "max_residue": max_res,
                "length": length,
            }
        else:
            logger.warning(f"Chain {chain_id} had no parseable ATOM residue numbers.")

    preview_str_parts = header_lines[: min(len(header_lines), preview_lines // 2)]
    remaining_preview_lines = preview_lines - len(preview_str_parts)
    preview_str_parts.extend(
        atom_lines[: min(len(atom_lines), remaining_preview_lines)]
    )
    pdb_preview = "\n".join(preview_str_parts)
    if len(pdb_content.splitlines()) > preview_lines:
        pdb_preview += "\n..."
    return chain_details, pdb_preview
