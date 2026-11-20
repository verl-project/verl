import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models.alphafold2 import call_alphafold2
from .models.alphafold2_multimer import call_alphafold2_multimer
from .models.proteinmpnn import call_proteinmpnn
from .models.rfdiffusion import call_rfdiffusion
from .utils.pdb_utils import get_pdb_chain_details

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        nim_api_key: str,
        api_timeout: int,
        polling_interval: int,
        output_dir: Path,
        debug_protein_design_calls: bool,
    ):
        self.nim_api_key = nim_api_key
        self.api_timeout = api_timeout
        self.polling_interval = polling_interval
        self.output_dir = output_dir
        self.debug_protein_design_calls = debug_protein_design_calls
        self._debug_af2m_call_count = 0

    def _validate_rfd_contigs(
        self, contigs_str: str, target_chain_details: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """
        Validates the RFDiffusion contigs string against target PDB chain details.
        Returns None if valid, or an error message string if invalid.
        """
        if not contigs_str:
            return "Contigs string is empty."

        target_segment_pattern = re.compile(r"([A-Za-z0-9])(\d+)-(\d+)")
        active_contig_parts = contigs_str.split(
            "/"
        )  # Split by binder definition markers

        for part in active_contig_parts:
            chain_segments_in_part = part.strip().split(" ")
            for segment_text in chain_segments_in_part:
                segment_text = segment_text.strip()
                if not segment_text or segment_text.isdigit():
                    continue

                match = target_segment_pattern.fullmatch(segment_text)
                if match:
                    seg_chain_id, seg_start_str, seg_end_str = match.groups()
                    seg_start = int(seg_start_str)
                    seg_end = int(seg_end_str)

                    if seg_chain_id not in target_chain_details:
                        return (
                            f"Chain '{seg_chain_id}' in contig segment '{segment_text}' not in target. "
                            f"Valid: {list(target_chain_details.keys())}."
                        )

                    chain_min = target_chain_details[seg_chain_id]["min_residue"]
                    chain_max = target_chain_details[seg_chain_id]["max_residue"]

                    if not (
                        chain_min <= seg_start <= chain_max
                        and chain_min <= seg_end <= chain_max
                        and seg_start <= seg_end
                    ):
                        return (
                            f"Residue range {seg_start}-{seg_end} for chain '{seg_chain_id}' in '{segment_text}' "
                            f"is invalid/out of bounds. Chain '{seg_chain_id}' actual range: {chain_min}-{chain_max}."
                        )
        return None

    def _validate_rfd_hotspots(
        self, hotspot_list: List[str], target_chain_details: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """
        Validates hotspot residues (e.g., ["A50", "B25"]) against target PDB chain details.
        Returns None if valid, or an error message string if invalid.
        """
        if not hotspot_list:
            return None

        hotspot_pattern = re.compile(r"([A-Za-z0-9])(\d+)")

        for hotspot_str in hotspot_list:
            match = hotspot_pattern.fullmatch(hotspot_str.strip())  # Add strip
            if not match:
                return (
                    f"Hotspot '{hotspot_str}' is not in expected format (e.g., 'A50')."
                )

            hs_chain_id, hs_res_num_str = match.groups()
            hs_res_num = int(hs_res_num_str)

            if hs_chain_id not in target_chain_details:
                return (
                    f"Chain '{hs_chain_id}' for hotspot '{hotspot_str}' not in target. "
                    f"Valid: {list(target_chain_details.keys())}."
                )

            chain_min = target_chain_details[hs_chain_id]["min_residue"]
            chain_max = target_chain_details[hs_chain_id]["max_residue"]

            if not (chain_min <= hs_res_num <= chain_max):
                return (
                    f"Residue {hs_res_num} for hotspot '{hotspot_str}' (chain '{hs_chain_id}') "
                    f"out of bounds. Chain '{hs_chain_id}' actual range: {chain_min}-{chain_max}."
                )
        return None

    async def _run_nim_alphafold2(self, args: Dict, workflow_state: Dict) -> Dict:
        """
        Runs AlphaFold2 for target structure prediction. Returns structured output with
        tool_output and state_updates separated.
        """
        item_id = workflow_state["item_id"]
        current_internal_step = workflow_state["current_internal_step"]
        target_sequence_from_state = workflow_state["target_sequence"]

        tool_output = {}
        state_updates = {}

        if self.debug_protein_design_calls:
            logger.warning(
                f"DEBUG MODE: Bypassing AlphaFold2 API call for workflow {item_id}."
            )
            module_dir = Path(__file__).parent
            fixed_pdb_path = module_dir / "debug_target.pdb"

            if not fixed_pdb_path.exists():
                logger.error(f"Debug mode failed: {fixed_pdb_path} not found.")
                tool_output = {
                    "success": False,
                    "error": f"Debug mode failed: Required file {fixed_pdb_path} not found.",
                }
                return {"tool_output": tool_output, "state_updates": state_updates}

            with open(fixed_pdb_path, "r") as f:
                pdb_content = f.read()

            chain_details, pdb_preview = get_pdb_chain_details(pdb_content)

            state_updates["target_pdb_content"] = pdb_content
            state_updates["target_chain_details"] = chain_details
            state_updates["target_pdb_preview"] = pdb_preview
            state_updates["target_structure_predicted"] = True

            debug_pdb_path = (
                self.output_dir
                / f"target_{item_id}_s{current_internal_step}_af2_DEBUG.pdb"
            )
            with open(debug_pdb_path, "w") as f:
                f.write(pdb_content)
            logger.info(f"DEBUG MODE: Copied fixed AlphaFold2 PDB to {debug_pdb_path}")

            tool_output = {
                "success": True,
                "message": "DEBUG MODE: Used fixed PDB for AlphaFold2.",
                "target_pdb_preview": pdb_preview,
                "saved_pdb_path": str(debug_pdb_path),
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        sequence_from_llm = args.get("sequence")
        if not sequence_from_llm:
            tool_output = {
                "success": False,
                "error": "Missing 'sequence' for AlphaFold2.",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        actual_sequence_to_use = target_sequence_from_state
        if sequence_from_llm != target_sequence_from_state:
            logger.warning(
                f"LLM provided sequence '{sequence_from_llm[:20]}...' for 'predict_target_structure_alphafold2'. "
                f"However, this tool will use the canonical target sequence from the workflow state: "
                f"'{target_sequence_from_state[:20]}...'"
            )

        api_result = await call_alphafold2(
            sequence=actual_sequence_to_use,
            api_key=self.nim_api_key,
            timeout=self.api_timeout,
            polling_interval=self.polling_interval,
        )
        if api_result and isinstance(api_result, list) and api_result[0]:
            pdb_content = api_result[0]
            chain_details, pdb_preview = get_pdb_chain_details(pdb_content)

            state_updates["target_pdb_content"] = pdb_content
            state_updates["target_chain_details"] = chain_details
            state_updates["target_pdb_preview"] = pdb_preview
            state_updates["target_structure_predicted"] = True

            pdb_path = (
                self.output_dir / f"target_{item_id}_s{current_internal_step}_af2.pdb"
            )
            with open(pdb_path, "w") as f:
                f.write(pdb_content)
            logger.info(
                f"Workflow {item_id}: AlphaFold2 PDB saved to {pdb_path}. "
                f"Chain details: {chain_details}"
            )

            tool_output = {
                "success": True,
                "message": "AlphaFold2 prediction complete.",
                "target_pdb_preview": pdb_preview,
                "saved_pdb_path": str(pdb_path),
            }
        else:
            error_detail = (
                api_result.get("error", "AlphaFold2 prediction failed.")
                if isinstance(api_result, dict)
                else "AlphaFold2 prediction failed."
            )
            logger.error(f"Workflow {item_id}: AlphaFold2 call failed: {error_detail}")
            tool_output = {"success": False, "error": error_detail}
            state_updates["target_structure_predicted"] = False

        return {"tool_output": tool_output, "state_updates": state_updates}

    async def _run_nim_rfdiffusion(self, args: Dict, workflow_state: Dict) -> Dict:
        """
        Runs RFDiffusion for binder backbone design. Returns structured output with
        tool_output and state_updates separated.
        """
        item_id = workflow_state["item_id"]
        current_internal_step = workflow_state["current_internal_step"]
        target_pdb_content = workflow_state.get("target_pdb_content")
        target_chain_details = workflow_state.get("target_chain_details", {})

        tool_output = {}
        state_updates = {}

        contigs_str_from_llm = args.get("contigs")
        if not target_pdb_content:
            tool_output = {
                "success": False,
                "error": "Target PDB not found in state for RFDiffusion.",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}
        if not contigs_str_from_llm:
            tool_output = {
                "success": False,
                "error": "Missing 'contigs' for RFDiffusion.",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        validation_error = self._validate_rfd_contigs(
            contigs_str_from_llm, target_chain_details
        )
        if validation_error:
            logger.warning(
                f"RFDiffusion contigs validation failed for item {item_id}: {validation_error}. "
                f"Contigs: '{contigs_str_from_llm}'"
            )
            tool_output = {
                "success": False,
                "error": f"Invalid contigs: {validation_error}",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        hotspot_residues = args.get("hotspot_residues")
        if hotspot_residues:
            hotspot_validation_error = self._validate_rfd_hotspots(
                hotspot_residues, target_chain_details
            )
            if hotspot_validation_error:
                logger.warning(
                    f"RFDiffusion hotspot validation failed for item {item_id}: {hotspot_validation_error}. "
                    f"Hotspots: {hotspot_residues}"
                )
                tool_output = {
                    "success": False,
                    "error": f"Invalid hotspots: {hotspot_validation_error}",
                }
                return {"tool_output": tool_output, "state_updates": state_updates}

        api_result = await call_rfdiffusion(
            input_pdb=target_pdb_content,
            api_key=self.nim_api_key,
            contigs=contigs_str_from_llm,
            hotspot_res=hotspot_residues,
            timeout=self.api_timeout,
            polling_interval=self.polling_interval,
        )

        if api_result and "output_pdb" in api_result:
            binder_pdb = api_result["output_pdb"]
            binder_chain_details, binder_pdb_preview = get_pdb_chain_details(binder_pdb)

            state_updates["binder_backbone_pdb_content"] = binder_pdb
            state_updates["binder_chain_details"] = binder_chain_details
            state_updates["binder_pdb_preview"] = binder_pdb_preview
            state_updates["binder_backbone_designed"] = True

            pdb_path = (
                self.output_dir
                / f"binder_backbone_{item_id}_s{current_internal_step}_rfd.pdb"
            )
            with open(pdb_path, "w") as f:
                f.write(binder_pdb)
            logger.info(f"Workflow {item_id}: RFDiffusion PDB saved to {pdb_path}")

            tool_output = {
                "success": True,
                "message": "RFDiffusion complete.",
                "binder_backbone_pdb_preview": binder_pdb_preview,
                "saved_pdb_path": str(pdb_path),
            }
        else:
            error_detail = (
                api_result.get("error", "RFDiffusion failed.")
                if isinstance(api_result, dict)
                else "RFDiffusion failed."
            )
            logger.error(
                f"Workflow {item_id}: RFDiffusion call failed: {error_detail}. API Result: {api_result}"
            )
            tool_output = {"success": False, "error": error_detail}
            state_updates["binder_backbone_designed"] = False

        return {"tool_output": tool_output, "state_updates": state_updates}

    async def _run_nim_proteinmpnn(self, args: Dict, workflow_state: Dict) -> Dict:
        """
        Runs ProteinMPNN for binder sequence design. Returns structured output with
        tool_output and state_updates separated.
        """
        item_id = workflow_state["item_id"]
        current_internal_step = workflow_state["current_internal_step"]
        binder_pdb = workflow_state.get("binder_backbone_pdb_content")

        tool_output = {}
        state_updates = {}

        if not binder_pdb:
            tool_output = {
                "success": False,
                "error": "Binder backbone PDB not found for ProteinMPNN.",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        sampling_temp_list = args.get("sampling_temp", [0.1])

        api_result = await call_proteinmpnn(
            input_pdb=binder_pdb,
            api_key=self.nim_api_key,
            sampling_temp=sampling_temp_list,
            timeout=self.api_timeout,
            polling_interval=self.polling_interval,
        )

        if not (api_result and "mfasta" in api_result):
            error_detail = (
                api_result.get(
                    "error", "ProteinMPNN call failed or no mfasta in result."
                )
                if isinstance(api_result, dict)
                else "PMPNN call failed"
            )
            logger.error(
                f"Workflow {item_id}: ProteinMPNN call failed: {error_detail}. API Result: {api_result}"
            )
            tool_output = {"success": False, "error": error_detail}
            state_updates["binder_sequence_designed"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        fasta_content = api_result["mfasta"]
        entries: List[Tuple[float, str, str]] = []
        current_header = None
        current_sequence_parts: List[str] = []
        for line_content in fasta_content.splitlines():
            line = line_content.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header and current_sequence_parts:
                    full_sequence_line = "".join(current_sequence_parts)
                    score_match = re.search(r"global_score=([-\d.]+)", current_header)
                    global_score = (
                        float(score_match.group(1)) if score_match else -float("inf")
                    )
                    entries.append((global_score, current_header, full_sequence_line))
                current_header = line
                current_sequence_parts = []
            else:
                current_sequence_parts.append(line)
        if current_header and current_sequence_parts:
            full_sequence_line = "".join(current_sequence_parts)
            score_match = re.search(r"global_score=([-\d.]+)", current_header)
            global_score = float(score_match.group(1)) if score_match else -float("inf")
            entries.append((global_score, current_header, full_sequence_line))

        if not entries:
            tool_output = {"success": False, "error": "No sequences parsed from PMPNN."}
            state_updates["binder_sequence_designed"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        entries.sort(key=lambda x: x[0], reverse=True)
        best_global_score, best_header, best_full_sequence_line = entries[0]
        logger.info(
            f"Workflow {item_id}: Best PMPNN sequence chosen (global_score={best_global_score:.4f}) "
            f"from header: '{best_header}' -> Seq line: '{best_full_sequence_line}'"
        )

        parsed_binder_chains = [
            s.strip() for s in best_full_sequence_line.split("/") if s.strip()
        ]

        if not parsed_binder_chains or not all(
            s and s.isalpha() and s.isupper() for s in parsed_binder_chains
        ):
            tool_output = {
                "success": False,
                "error": (
                    f"Invalid binder chains from PMPNN after parsing '{best_full_sequence_line}'. "
                    f"Parsed: {parsed_binder_chains}"
                ),
            }
            state_updates["binder_sequence_designed"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        state_updates["designed_binder_sequence"] = parsed_binder_chains
        state_updates["binder_sequence_designed"] = True

        fasta_path = (
            self.output_dir
            / f"binder_sequence_{item_id}_s{current_internal_step}_pmpnn.fasta"
        )
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        logger.info(
            f"Workflow {item_id}: ProteinMPNN FASTA saved to {fasta_path}. "
            f"Selected binder chains: {parsed_binder_chains}"
        )

        preview = (
            parsed_binder_chains[0][:60] + "..." if parsed_binder_chains else "N/A"
        )
        if len(parsed_binder_chains) > 1:
            preview += f" (+ {len(parsed_binder_chains)-1} more chain(s))"

        tool_output = {
            "success": True,
            "message": (
                f"ProteinMPNN complete. Selected best (global_score={best_global_score:.4f})."
            ),
            "designed_binder_sequence_list": parsed_binder_chains,
            "designed_binder_sequence_preview": preview,
            "saved_fasta_path": str(fasta_path),
        }
        return {"tool_output": tool_output, "state_updates": state_updates}

    async def _run_nim_af2_multimer(self, args: Dict, workflow_state: Dict) -> Dict:
        item_id = workflow_state["item_id"]
        current_internal_step = workflow_state["current_internal_step"]
        target_seq = workflow_state.get("target_sequence")
        designed_binder_chains_list = workflow_state.get("designed_binder_sequence")

        tool_output = {}
        state_updates = {}

        if (
            not target_seq
            or not designed_binder_chains_list
            or not isinstance(designed_binder_chains_list, list)
        ):
            tool_output = {
                "success": False,
                "error": "Missing or invalid sequences for AF2-Multimer.",
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        all_input_sequences_for_multimer = [target_seq] + designed_binder_chains_list

        for i, seq_to_validate in enumerate(all_input_sequences_for_multimer):
            if not (
                seq_to_validate
                and isinstance(seq_to_validate, str)
                and seq_to_validate.isalpha()
                and seq_to_validate.isupper()
            ):
                error_msg = (
                    f"Sequence {i+1} (part of target/binder complex) is invalid: "
                    f"'{str(seq_to_validate)[:30]}...'. "
                    f"Contains non-alpha/lowercase, is empty, or not a string."
                )
                logger.error(f"Workflow {item_id}: {error_msg}")
                tool_output = {"success": False, "error": error_msg}
                return {"tool_output": tool_output, "state_updates": state_updates}

        relax = args.get("relax_prediction", True)

        if self.debug_protein_design_calls:
            self._debug_af2m_call_count += 1
            mock_plddt = 87.5 if self._debug_af2m_call_count % 2 == 1 else 45.2
            success_message = (
                f"DEBUG MODE: Returning {'high' if mock_plddt > 50 else 'low'}-quality "
                f"mock results (call #{self._debug_af2m_call_count})"
            )

            debug_pdb_filename = f"complex_{item_id}_s{current_internal_step}_af2m_DEBUG_pLDDT{mock_plddt:.2f}.pdb"
            debug_pdb_path = self.output_dir / debug_pdb_filename
            try:
                with open(debug_pdb_path, "w") as f:
                    f.write(
                        f"REMARK DEBUG PDB FILE for complex. Predicted pLDDT {mock_plddt}\n"
                    )
                logger.info(
                    f"DEBUG MODE: Saved mock AF2-Multimer PDB to {debug_pdb_path}"
                )
                state_updates["complex_pdb_content_path"] = str(debug_pdb_path)
            except IOError as e:
                logger.error(
                    f"DEBUG MODE: Failed to write mock PDB {debug_pdb_path}: {e}"
                )
                # If saving fails, don't set the path, but can still proceed with mock pLDDT
                state_updates["complex_pdb_content_path"] = None

            state_updates["af2_multimer_plddt"] = mock_plddt
            state_updates["complex_evaluated"] = True

            tool_output = {
                "success": True,
                "message": f"{success_message}. Mock pLDDT: {mock_plddt:.2f}",
                "plddt": mock_plddt,
                "complex_file_path": (
                    str(debug_pdb_path)
                    if state_updates["complex_pdb_content_path"]
                    else None
                ),
            }
            return {"tool_output": tool_output, "state_updates": state_updates}

        api_result = await call_alphafold2_multimer(
            sequences=all_input_sequences_for_multimer,
            api_key=self.nim_api_key,
            relax_prediction=relax,
            timeout=self.api_timeout,
            polling_interval=self.polling_interval,
        )

        if api_result is None or (
            isinstance(api_result, dict) and api_result.get("success") is False
        ):
            error_detail = "AF2-Multimer call failed or returned None."
            if isinstance(api_result, dict):
                error_detail = api_result.get(
                    "error", "AF2-Multimer call failed with unspecified error."
                )
                detail_info = api_result.get("detail", "")
                if detail_info:
                    error_detail += f" Details: {detail_info}"

            logger.error(
                f"Workflow {item_id}: AF2-Multimer call failed: {error_detail}. "
                f"API Result: {api_result}"
            )
            tool_output = {"success": False, "error": error_detail}
            state_updates["complex_evaluated"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        all_structures_info = api_result.get("structures")
        if not all_structures_info or not isinstance(all_structures_info, list):
            message = api_result.get(
                "message", "No structures returned from AF2-Multimer process."
            )
            logger.warning(f"Workflow {item_id}: {message}. API Result: {api_result}")
            if not all_structures_info and isinstance(all_structures_info, list):
                tool_output = {
                    "success": True,
                    "message": (
                        "AF2-Multimer ran, but no PDB structures were produced by the API."
                    ),
                    "plddt": 0.0,
                    "complex_file_path": None,
                }
                state_updates["af2_multimer_plddt"] = 0.0
                state_updates["complex_evaluated"] = True
                state_updates["complex_pdb_content_path"] = None
            else:
                tool_output = {
                    "success": False,
                    "error": (
                        "AF2-Multimer returned unexpected data or no structures."
                    ),
                }
                state_updates["complex_evaluated"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        best_structure_info = None
        highest_plddt = -1.0

        for struct_info in all_structures_info:
            current_plddt = struct_info.get("average_plddt", 0.0)
            if current_plddt > highest_plddt:
                highest_plddt = current_plddt
                best_structure_info = struct_info

        if best_structure_info is None:
            logger.error(
                f"Workflow {item_id}: No valid structure with pLDDT found in AF2-Multimer results, "
                f"though structures were present."
            )
            tool_output = {
                "success": False,
                "error": ("No valid structure with pLDDT in AF2-Multimer results."),
            }
            state_updates["complex_evaluated"] = False
            return {"tool_output": tool_output, "state_updates": state_updates}

        best_pdb_content = best_structure_info.get("pdb_content")
        best_plddt = best_structure_info.get("average_plddt", 0.0)
        best_model_idx = best_structure_info.get("model_index", "NA")

        if not best_pdb_content:
            logger.error(
                f"Workflow {item_id}: Best AF2-Multimer structure (Model {best_model_idx}, "
                f"pLDDT {best_plddt:.2f}) found, but PDB content is missing."
            )
            tool_output = {
                "success": False,
                "error": f"Best model (pLDDT {best_plddt:.2f}) has no PDB content.",
            }
            state_updates["complex_evaluated"] = False
            state_updates["af2_multimer_plddt"] = best_plddt
            return {"tool_output": tool_output, "state_updates": state_updates}

        complex_pdb_filename = (
            f"complex_{item_id}_s{current_internal_step}_af2m_model{best_model_idx}_"
            f"pLDDT{best_plddt:.2f}.pdb"
        )
        complex_pdb_path = self.output_dir / complex_pdb_filename

        try:
            with open(complex_pdb_path, "w", encoding="utf-8") as f:
                f.write(best_pdb_content)
            logger.info(
                f"Workflow {item_id}: AlphaFold2-Multimer complete. Saved best model (Index {best_model_idx}) "
                f"with pLDDT: {best_plddt:.2f} from {len(all_structures_info)} models to {complex_pdb_path}"
            )

            state_updates["complex_pdb_content_path"] = str(complex_pdb_path)
            state_updates["af2_multimer_plddt"] = best_plddt
            state_updates["complex_evaluated"] = True

            complex_quality_message = (
                f"AlphaFold2-Multimer evaluation complete. Selected best model (Index {best_model_idx}) "
                f"with pLDDT: {best_plddt:.2f}"
            )

            tool_output = {
                "success": True,
                "message": complex_quality_message,
                "plddt": best_plddt,
                "complex_file_path": str(complex_pdb_path),
                "selected_model_index": best_model_idx,
            }
        except IOError as e:
            logger.error(
                f"Workflow {item_id}: Failed to save best AF2-Multimer PDB (Model {best_model_idx}, "
                f"pLDDT {best_plddt:.2f}) to {complex_pdb_path}: {e}"
            )
            tool_output = {
                "success": False,
                "error": f"Failed to save best complex PDB: {e}",
            }
            state_updates["af2_multimer_plddt"] = best_plddt
            state_updates["complex_pdb_content_path"] = None
            state_updates["complex_evaluated"] = True

        return {"tool_output": tool_output, "state_updates": state_updates}

    async def dispatch_tool_call(
        self, tool_name: str, args: Dict, workflow_state: Dict
    ) -> Dict:
        """Main dispatch method for executing tools."""
        item_id = workflow_state["item_id"]
        internal_step = workflow_state["current_internal_step"]
        logger.info(
            f"ToolExecutor: Dispatching tool '{tool_name}' for workflow {item_id}, "
            f"Step {internal_step} with args: {args}"
        )

        if not self.nim_api_key:
            return {
                "tool_output": {
                    "success": False,
                    "error": "NIM API key not configured in ToolExecutor.",
                },
                "state_updates": {},
            }

        if tool_name == "predict_target_structure_alphafold2":
            return await self._run_nim_alphafold2(args, workflow_state)
        elif tool_name == "design_binder_backbone_rfdiffusion":
            return await self._run_nim_rfdiffusion(args, workflow_state)
        elif tool_name == "design_binder_sequence_proteinmpnn":
            return await self._run_nim_proteinmpnn(args, workflow_state)
        elif tool_name == "evaluate_binder_complex_alphafold2_multimer":
            return await self._run_nim_af2_multimer(args, workflow_state)
        else:
            logger.error(
                f"ToolExecutor: Unknown tool name '{tool_name}' for workflow {item_id}"
            )
            return {
                "tool_output": {
                    "success": False,
                    "error": f"Unknown tool name: {tool_name}",
                },
                "state_updates": {},
            }
