import argparse
import asyncio
import json
import math
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pybullet as p
import pybullet_data
import wandb
import websockets

# LLM Service Import
from .llm_services import get_anthropic_completion


@dataclass
class ObjectState:
    id: str
    type: str  # 'cube', 'sphere'
    position: List[float]
    orientation_quaternion: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color_rgba: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])


@dataclass
class SpatialTask:
    task_id: str
    description: str
    initial_objects: List[ObjectState]
    goal_description: str
    target_object_id: str
    reference_object_id: str
    target_distance: float = 1.0


class MVPPhysicsSimulator:
    def __init__(self):
        self.client_id = -1
        self.objects_pb_ids: Dict[str, int] = {}
        self.object_configs: Dict[str, ObjectState] = {}

    def initialize(self, objects: List[ObjectState]):
        if self.client_id != -1:
            p.disconnect(physicsClientId=self.client_id)

        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        self.objects_pb_ids = {}
        self.object_configs = {}
        for obj_state in objects:
            self._add_object(obj_state)
        print(f"Physics initialized with {len(self.objects_pb_ids)} objects.")

    def _add_object(self, obj_state: ObjectState):
        half_extents = [s / 2.0 for s in obj_state.scale]
        shape_id = -1
        if obj_state.type == "cube":
            shape_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
            )
        elif obj_state.type == "sphere":
            shape_id = p.createCollisionShape(
                p.GEOM_SPHERE, radius=half_extents[0], physicsClientId=self.client_id
            )
        else:
            print(
                f"Warning: Unsupported object type '{obj_state.type}' for object ID '{obj_state.id}'"
            )
            return

        if obj_state.type == "cube":
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=obj_state.color_rgba,
                physicsClientId=self.client_id,
            )
        elif obj_state.type == "sphere":
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=half_extents[0],
                rgbaColor=obj_state.color_rgba,
                physicsClientId=self.client_id,
            )
        else:
            print(
                f"Warning: Unsupported object type '{obj_state.type}' for object ID '{obj_state.id}'"
            )
            return

        body_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=obj_state.position,
            baseOrientation=obj_state.orientation_quaternion,
            physicsClientId=self.client_id,
        )
        self.objects_pb_ids[obj_state.id] = body_id
        self.object_configs[obj_state.id] = obj_state

    def move_object(
        self,
        object_id: str,
        target_position: List[float],
        target_orientation_quaternion: Optional[List[float]] = None,
    ):
        if object_id in self.objects_pb_ids:
            body_id = self.objects_pb_ids[object_id]
            if target_orientation_quaternion is None:
                _, current_orientation = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self.client_id
                )
                target_orientation_quaternion = list(current_orientation)
            p.resetBasePositionAndOrientation(
                body_id,
                target_position,
                target_orientation_quaternion,
                physicsClientId=self.client_id,
            )
        else:
            print(f"Warning: Attempted to move unknown object ID '{object_id}'")

    def simulate_steps(self, steps: int = 10):
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_current_state_for_visualization(self) -> List[Dict[str, Any]]:
        viz_state = []
        for obj_id, body_id in self.objects_pb_ids.items():
            pos, orn_quat = p.getBasePositionAndOrientation(
                body_id, physicsClientId=self.client_id
            )
            original_config = self.object_configs.get(obj_id)
            if original_config:
                viz_state.append(
                    {
                        "id": obj_id,
                        "type": original_config.type,
                        "position": list(pos),
                        "orientation_quaternion": list(orn_quat),
                        "scale": original_config.scale,
                        "color_rgba": original_config.color_rgba,
                    }
                )
        return viz_state

    def calculate_distance(self, obj1_id: str, obj2_id: str) -> float:
        pos1, pos2 = None, None
        current_state = self.get_current_state_for_visualization()
        for obj_data in current_state:
            if obj_data["id"] == obj1_id:
                pos1 = obj_data["position"]
            if obj_data["id"] == obj2_id:
                pos2 = obj_data["position"]

        if pos1 and pos2:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
        return float("inf")

    def cleanup(self):
        if self.client_id != -1:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1
        print("Physics simulation cleaned up.")


connected_visualization_clients = set()
global_physics_simulator_instance: Optional[MVPPhysicsSimulator] = None

# To make demo_runner accessible to the WebSocket handler in a cleaner way for server_mode
shared_demo_runner_instance: Optional["MVPDemoRunner"] = None


async def notify_visualization_clients(scene_state: List[Dict[str, Any]]):
    if connected_visualization_clients:
        message = json.dumps({"type": "scene_update", "payload": scene_state})
        await asyncio.gather(
            *[client.send(message) for client in connected_visualization_clients]
        )


async def visualization_websocket_handler(websocket):
    global global_physics_simulator_instance, shared_demo_runner_instance  # Make shared_demo_runner_instance accessible
    connected_visualization_clients.add(websocket)
    print(
        f"Visualization client connected: {websocket.remote_address} (Total: {len(connected_visualization_clients)})"
    )
    try:
        if global_physics_simulator_instance:
            initial_state = (
                global_physics_simulator_instance.get_current_state_for_visualization()
            )
            await websocket.send(
                json.dumps({"type": "initial_scene", "payload": initial_state})
            )

        async for message_str in websocket:
            print(f"Message from viz client: {message_str}")
            try:
                data = json.loads(message_str)
                command = data.get("command")

                if command == "next_llm_task":
                    print("Received 'next_llm_task' command from client.")
                    if shared_demo_runner_instance:
                        # Run a single turn, which now defaults to using the real LLM via llm_services
                        asyncio.create_task(
                            shared_demo_runner_instance.run_single_turn_demo(
                                use_real_llm=True
                            )
                        )
                    else:
                        print(
                            "Error: shared_demo_runner_instance not found to execute 'next_llm_task'"
                        )
                else:
                    print(f"Unknown command received: {command}")

            except json.JSONDecodeError:
                print(f"Invalid JSON from client: {message_str}")
            except Exception as e:
                print(f"Error processing client command: {e}")
    except websockets.exceptions.ConnectionClosed:
        print(
            f"Visualization client disconnected. (Total: {len(connected_visualization_clients)-1})"
        )
    except Exception as e:
        print(f"Error in visualization_websocket_handler: {e}")
    finally:
        connected_visualization_clients.remove(websocket)


class SpatialEnvironmentMVP:
    def __init__(self):
        global global_physics_simulator_instance
        self.simulator = MVPPhysicsSimulator()
        global_physics_simulator_instance = self.simulator
        self.current_task: Optional[SpatialTask] = None
        self.task_id_counter = 0

    async def get_next_item(self) -> Dict[str, Any]:
        self.task_id_counter += 1
        task_id = f"conditional_task_{self.task_id_counter}_{uuid.uuid4().hex[:4]}"

        # Start objects on opposite sides, e.g., along the x-axis
        objects = [
            ObjectState(
                id="red_cube",
                type="cube",
                position=[2.0, 0.5, 0.5],
                scale=[1, 1, 1],
                color_rgba=[1, 0, 0, 1],
            ),
            ObjectState(
                id="blue_sphere",
                type="sphere",
                position=[-2.0, 0.5, 0.5],
                scale=[1, 1, 1],
                color_rgba=[0, 0, 1, 1],
            ),
        ]

        task_description = (
            "The red cube and blue sphere are on opposite sides of the YZ plane (different X signs). "
            "Move the red cube so it remains on the opposite side of the YZ plane from the blue sphere, "
            "but position it very close to the blue sphere (approximately 1.0 unit away)."
        )
        goal_description = (
            "The red_cube's final x-coordinate should have the opposite sign to the blue_sphere's x-coordinate. "
            "The distance between the center of the red_cube and the center of the blue_sphere "
            "should be approximately 1.0 unit."
        )

        task = SpatialTask(
            task_id=task_id,
            description=task_description,
            initial_objects=objects,
            goal_description=goal_description,
            target_object_id="red_cube",
            reference_object_id="blue_sphere",
            target_distance=1.0,  # This remains the target for proximity
        )
        self.current_task = task
        self.simulator.initialize(task.initial_objects)

        await notify_visualization_clients(
            self.simulator.get_current_state_for_visualization()
        )

        return {
            "task_id": task.task_id,
            "llm_prompt": self._create_llm_prompt(
                task, objects
            ),  # Pass initial objects for the prompt
        }

    def _create_llm_prompt(
        self, task: SpatialTask, initial_objects_state: List[ObjectState]
    ) -> str:
        # Use the passed initial_objects_state for accurate current positions in the prompt
        objects_desc_parts = []
        for obj_state in initial_objects_state:
            # Find the current position from the simulator if it has been initialized and objects added
            # For the initial prompt, obj_state.position IS the current position.
            objects_desc_parts.append(
                f"- ID: {obj_state.id}, Type: {obj_state.type}, "
                f"Current Position: [{obj_state.position[0]:.2f}, {obj_state.position[1]:.2f}, "
                f"{obj_state.position[2]:.2f}]"
            )
        objects_desc = "\n".join(objects_desc_parts)

        # Reference object's current position for the hint
        ref_obj_pos_str = "N/A"
        for obj_state in initial_objects_state:
            if obj_state.id == task.reference_object_id:
                ref_obj_pos_str = (
                    f"[{obj_state.position[0]:.2f}, {obj_state.position[1]:.2f}, "
                    f"{obj_state.position[2]:.2f}]"
                )
                break

        hint = (
            f"Hint: The blue_sphere (reference object) is currently at {ref_obj_pos_str}. "
            f"To keep the red_cube on the opposite side of the YZ plane, its x-coordinate "
            f"should generally have the opposite sign to the blue_sphere's x-coordinate. "
            f"Adjust its position to be about {task.target_distance:.1f} unit away from the blue_sphere."
        )

        return f"""Task: {task.description}
Goal: {task.goal_description}

Available Objects (initial state):
{objects_desc}

{hint}

You control: '{task.target_object_id}'.
Your action MUST be a JSON object like:
{{
    "action_type": "move_object",
    "object_id": "{task.target_object_id}",
    "target_position": [x_float, y_float, z_float]  # New target coordinates
}}
Only provide the JSON for the action. Do not add any other text or explanations.
Your JSON action:"""

    async def collect_trajectories(
        self, item_from_get_next: Dict[str, Any], llm_completion_raw: str
    ) -> Dict[str, Any]:
        if not self.current_task:
            return {
                "error": "No current task set. Call get_next_item first.",
                "score": 0.0,
            }

        llm_prompt_for_api = item_from_get_next["llm_prompt"]

        print(
            "\nDEBUG SPATIAL_ENV: Calling get_anthropic_completion with prompt... Timeout in 30s"
        )
        try:
            # Add a timeout to the LLM call to prevent indefinite hanging
            llm_completion_raw = await asyncio.wait_for(
                get_anthropic_completion(llm_prompt_for_api), timeout=30.0
            )
        except asyncio.TimeoutError:
            print(
                "DEBUG SPATIAL_ENV: LLM call timed out after 30s. Using fallback mock."
            )
            llm_completion_raw = None  # Indicate timeout
        except Exception as e:
            print(
                f"DEBUG SPATIAL_ENV: Error during get_anthropic_completion call: {e}. Using fallback mock."
            )
            llm_completion_raw = None  # Indicate other error

        print(
            f"DEBUG SPATIAL_ENV: llm_completion_raw received from get_anthropic_completion: '{llm_completion_raw}'"
        )

        parsed_action = None
        # Check if llm_completion_raw is valid before attempting to parse
        if (
            not llm_completion_raw
            or not isinstance(llm_completion_raw, str)
            or llm_completion_raw.strip() == ""
            or llm_completion_raw.strip() == "."
        ):
            print(
                f"DEBUG SPATIAL_ENV: llm_completion_raw is invalid ('{llm_completion_raw}'). "
                f"Using internal mock action."
            )
            # Define a valid mock action string here
            internal_mock_action_dict = {
                "action_type": "move_object",
                "object_id": self.current_task.target_object_id,
                "target_position": [1.0, 0.5, 0.5],
            }  # Example mock
            llm_completion_raw = json.dumps(
                internal_mock_action_dict
            )  # Ensure it's a JSON string for parsing
            print(f"DEBUG SPATIAL_ENV: Substituted internal mock: {llm_completion_raw}")

        try:
            json_str = llm_completion_raw.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            action_data = json.loads(json_str)
            if (
                action_data.get("action_type") == "move_object"
                and action_data.get("object_id") == self.current_task.target_object_id
                and isinstance(action_data.get("target_position"), list)
                and len(action_data.get("target_position")) == 3
            ):
                parsed_action = action_data
            else:
                print(
                    f"Warning: LLM action malformed or targets wrong object: {action_data}"
                )
        except json.JSONDecodeError as e:
            print(
                f"Warning: LLM response not valid JSON: {llm_completion_raw}. Error: {e}"
            )
        except Exception as e:
            print(
                f"Warning: Unexpected error parsing LLM response: {e}. Response: {llm_completion_raw}"
            )

        if parsed_action:
            self.simulator.move_object(
                object_id=parsed_action["object_id"],
                target_position=parsed_action["target_position"],
            )
            self.simulator.simulate_steps(20)
            await notify_visualization_clients(
                self.simulator.get_current_state_for_visualization()
            )
            await asyncio.sleep(0.05)
        else:
            print("No valid action parsed or executed. Scoring based on current state.")
            self.simulator.simulate_steps(5)
            await notify_visualization_clients(
                self.simulator.get_current_state_for_visualization()
            )

        distance = self.simulator.calculate_distance(
            self.current_task.target_object_id, self.current_task.reference_object_id
        )

        initial_ref_pos = None
        # Find the initial position of the reference object (blue_sphere) from the stored task data
        for obj_state in self.current_task.initial_objects:
            if obj_state.id == self.current_task.reference_object_id:
                initial_ref_pos = obj_state.position
                break

        final_target_pos = None
        # final_sim_state_viz is needed for metadata anyway, and has current positions
        final_sim_state_viz = self.simulator.get_current_state_for_visualization()
        for obj_data in final_sim_state_viz:
            if obj_data["id"] == self.current_task.target_object_id:
                final_target_pos = obj_data["position"]
                break

        side_condition_met = False
        if initial_ref_pos and final_target_pos:
            # Task: red cube (target) starts at x=2, blue sphere (ref) at x=-2.
            # Goal: move red cube near blue sphere. It should end up with x < 0 (same side as blue sphere).
            initial_ref_x_sign = (
                math.copysign(1.0, initial_ref_pos[0])
                if initial_ref_pos[0] != 0
                else 0.0
            )
            final_target_x_sign = (
                math.copysign(1.0, final_target_pos[0])
                if final_target_pos[0] != 0
                else 0.0
            )

            if initial_ref_x_sign != 0:  # Avoid issues if ref_obj starts at x=0
                side_condition_met = final_target_x_sign == initial_ref_x_sign
            else:
                side_condition_met = (
                    abs(final_target_pos[0]) < 0.5
                )  # If ref is at x=0, target should also be near x=0
            print(
                f"DEBUG SCORING: InitialRefXSign: {initial_ref_x_sign}, "
                f"FinalTargetXSign: {final_target_x_sign}, SideConditionMet: {side_condition_met}"
            )
        else:
            print(
                "DEBUG SCORING: Could not determine initial/final positions for side condition check."
            )

        score = 0.0
        # Max 0.8 points for distance
        if distance <= self.current_task.target_distance:  # e.g., <= 1.0
            score = 0.8
        elif (
            distance <= self.current_task.target_distance * 1.25
        ):  # More lenient threshold for high score band
            score = 0.6
        elif distance <= self.current_task.target_distance * 1.75:
            score = 0.4
        elif distance <= self.current_task.target_distance * 2.5:
            score = 0.2

        # Bonus 0.2 points for correct side condition
        if (
            side_condition_met
        ):  # Give bonus if side condition is met, regardless of exact distance score (as long as it tried)
            score += 0.2

        score = round(min(score, 1.0), 2)  # Cap score at 1.0 and round

        return {
            "request_id": self.current_task.task_id,
            "prompt_used": item_from_get_next["llm_prompt"],
            "llm_completion_raw": llm_completion_raw,
            "parsed_action": parsed_action,
            "score": score,
            "metadata": {
                "task_description": self.current_task.description,
                "final_distance": round(distance, 2),
                "target_distance": self.current_task.target_distance,
                "side_condition_met": side_condition_met,
                "final_sim_state_viz": final_sim_state_viz,
            },
        }


class MVPDemoRunner:
    def __init__(self):
        self.env = SpatialEnvironmentMVP()

    async def run_single_turn_demo(self, use_real_llm: bool = True):
        print("\n--- Running MVP Demo Turn ---")
        next_item_data = await self.env.get_next_item()
        task_id = next_item_data["task_id"]
        print(f"Task ID: {task_id}")
        # LLM Prompt is now printed by the llm_service if use_real_llm is True, or before collect_trajectories if not.
        # print(f"LLM Prompt:\n{llm_prompt}")

        # The llm_completion argument to collect_trajectories is now effectively ignored if use_real_llm is true,
        # as collect_trajectories will call the LLM service itself.
        # For mock behavior when use_real_llm is False, we might need to adjust.
        # However, our llm_service has its own mock, so we can rely on that for now if API key is missing.

        # For clarity, if we are NOT using real LLM (e.g. for process mode without API key),
        # we should generate a mock completion here and pass it.
        # But since llm_services.py has a fallback, we might not need a separate mock here IF
        # the intention is for collect_trajectories to ALWAYS try the LLM service path.
        # Let's assume collect_trajectories now always drives the LLM call.

        # The llm_completion parameter for collect_trajectories is now mostly for the original mock structure.
        # We can pass an empty string or None, as it will be replaced by the real LLM call internally.
        result = await self.env.collect_trajectories(
            next_item_data, ""
        )  # Pass dummy llm_completion

        print(f"\n--- Result for Task {task_id} ---")
        print(f"Final Score: {result['score']:.2f}")
        print(
            f"Achieved Distance: {result['metadata']['final_distance']:.2f} "
            f"(Target: {result['metadata']['target_distance']:.2f})"
        )
        return result


async def process_mode(args):
    print(
        f"Running in 'process' mode: generating {args.num_turns} trajectories to {args.output_file}"
    )

    run_name = f"padres_process_{args.num_turns}turns_{uuid.uuid4().hex[:4]}"
    wandb_is_initialized = False
    try:
        wandb.init(
            project="nous_hackathon_padres",  # Project name for W&B
            name=run_name,
            config=vars(args),  # Log command line arguments
        )
        print(f"W&B Run initialized: {run_name}. View at: {wandb.run.get_url()}")
        wandb_is_initialized = True
    except Exception as e:
        print(f"W&B initialization failed: {e}. Proceeding without W&B logging.")
        # Optionally, initialize in disabled mode: wandb.init(mode="disabled")

    demo_runner = MVPDemoRunner()
    results_to_write = []

    try:
        for i in range(args.num_turns):
            turn_num = i + 1
            print(f"\n--- Generating Trajectory Turn {turn_num}/{args.num_turns} ---")
            turn_result = (
                await demo_runner.run_single_turn_demo()
            )  # Assumes run_single_turn_demo uses real LLM by default now
            results_to_write.append(turn_result)

            if wandb_is_initialized and wandb.run:
                wandb_log_data = {
                    "turn": turn_num,
                    "task_id": turn_result.get("request_id", "N/A"),
                    "score": turn_result.get("score", 0.0),
                    "final_distance": turn_result.get("metadata", {}).get(
                        "final_distance", float("inf")
                    ),
                    "target_distance": turn_result.get("metadata", {}).get(
                        "target_distance", 0.0
                    ),
                    "side_condition_met": int(
                        turn_result.get("metadata", {}).get("side_condition_met", False)
                    ),
                }
                if turn_result.get("parsed_action"):
                    parsed_action = turn_result["parsed_action"]
                    wandb_log_data["action_object_id"] = parsed_action.get("object_id")
                    target_pos = parsed_action.get(
                        "target_position", [None, None, None]
                    )
                    wandb_log_data["action_target_x"] = (
                        target_pos[0] if target_pos and len(target_pos) > 0 else None
                    )
                    wandb_log_data["action_target_y"] = (
                        target_pos[1] if target_pos and len(target_pos) > 1 else None
                    )
                    wandb_log_data["action_target_z"] = (
                        target_pos[2] if target_pos and len(target_pos) > 2 else None
                    )

                wandb.log(wandb_log_data)
                print(
                    f"Logged to W&B: Turn {turn_num}, Score: {turn_result.get('score')}"
                )

            await asyncio.sleep(0.1)

        with open(args.output_file, "w") as f:
            for result_item in results_to_write:
                f.write(json.dumps(result_item) + "\n")
        print(
            f"\nSuccessfully wrote {len(results_to_write)} trajectories to {args.output_file}"
        )

    finally:
        if demo_runner.env.simulator:
            demo_runner.env.simulator.cleanup()
        if wandb_is_initialized and wandb.run:
            wandb.finish()
            print("Processing complete. W&B run finished.")
        else:
            print(
                "Processing complete. (W&B was not fully initialized or did not start a run)"
            )


async def server_mode():
    global shared_demo_runner_instance  # Make demo_runner available to the handler via this global
    shared_demo_runner_instance = MVPDemoRunner()

    websocket_server = await websockets.serve(
        visualization_websocket_handler,  # The handler will use shared_demo_runner_instance
        "localhost",
        8765,
    )
    print("Visualization WebSocket Server started on ws://localhost:8765")
    print("Open visualization/index.html in your browser.")
    print("You can run multiple demo turns. Press Ctrl+C to stop everything.")

    try:
        # Default 5 auto turns in server mode, now using LLM by default
        for i in range(5):  # Changed from 3 to 5
            print(f"\n--- Auto Demo Turn {i+1} ---")
            # use_real_llm=True by default
            await shared_demo_runner_instance.run_single_turn_demo(use_real_llm=True)
            await asyncio.sleep(2)

        print(
            "\nAutomatic demo turns complete. Server is still running for manual interaction or further tests."
        )
        await websocket_server.wait_closed()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        websocket_server.close()
        await websocket_server.wait_closed()
        if shared_demo_runner_instance.env.simulator:
            shared_demo_runner_instance.env.simulator.cleanup()
        print("Servers and physics simulation stopped.")


async def main():
    parser = argparse.ArgumentParser(description="Spatial RL Environment MVP")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    process_parser = subparsers.add_parser("process", help="Generate trajectory data")
    process_parser.add_argument(
        "--num_turns",
        type=int,
        default=5,
        help="Number of trajectory turns to generate",
    )
    process_parser.add_argument(
        "--output_file",
        type=str,
        default="trajectories.jsonl",
        help="File to save trajectory data",
    )

    args, unknown = parser.parse_known_args()

    if args.command == "process":
        await process_mode(args)
    elif args.command is None and not unknown:
        print("No command specified, running in default server mode.")
        await server_mode()
    elif unknown:
        print(f"Unknown arguments or command: {unknown}")
        parser.print_help()
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
