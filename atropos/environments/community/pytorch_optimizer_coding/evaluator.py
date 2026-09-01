from dotenv import load_dotenv
from verdict import Layer, Pipeline
from verdict.common.judge import CategoricalJudgeUnit, JudgeUnit
from verdict.scale import ContinuousScale, DiscreteScale
from verdict.schema import Schema
from verdict.transform import MaxPoolUnit

load_dotenv()


class OptimizerEvaluator:
    def __init__(self):
        self.score_pipeline = (
            Pipeline()
            >> Layer(
                JudgeUnit(scale=ContinuousScale(1, 10)).prompt(
                    (
                        "You are a judge that is an expert at evaluating optimizers for their novelty "
                        "as they will be accepted to a prestigious research conference. Given the following "
                        "optimizer code and its architecture/use-case, you must rate it on a scale of 1 to 10 "
                        "based on how novel it is and its impactfulness in speeding up model training. "
                        "Here is the code: {source.optimizer_code}\n"
                        "Here is the architecture: {source.architecture}"
                    )
                ),
                repeat=3,
            ).via("xai/grok-3-latest")
            >> MaxPoolUnit()
        )

        self.validity_pipeline = Pipeline() >> Layer(
            CategoricalJudgeUnit(
                name="Judge",
                categories=DiscreteScale(["yes", "no"]),
                explanation=False,
            )
            .prompt(
                """
                    You are an expert code validator specializing in PyTorch optimizers.
                    Your task is to determine if the provided optimizer code is completely valid and error-free.

                    A valid optimizer MUST satisfy ALL of these criteria:
                    1. Has zero syntax or runtime errors:
                       - No undefined variables
                       - No type mismatches
                       - No memory issues
                       - No CUDA/CPU compatibility problems
                    2. Can be imported and instantiated without blocking errors
                    3. Can run a complete optimization step without exceptions

                    Optimizer Code: {source.optimizer_code}
                    Stdout: {source.stdout}
                    Stderr: {source.stderr}

                    Respond with:
                    - "yes" if ALL criteria are met and the code is completely error-free
                    - "no" if ANY criterion fails or there are ANY potential issues

                    Be extremely strict in your evaluation.
                """
            )
            .via("xai/grok-3-latest", retries=2)
        )

    def score(self, optimizer_code: str, architecture: str) -> int:
        schema = Schema.of(
            optimizer_code=optimizer_code,
            architecture=architecture,
        )
        response, _ = self.score_pipeline.run(schema)
        return response.get("Pipeline_root.block.block.unit[Map MaxPool]_score", 0.0)

    def check_validity(self, optimizer_code: str, stdout: str, stderr: str) -> bool:
        schema = Schema.of(
            optimizer_code=optimizer_code,
            stdout=stdout,
            stderr=stderr,
        )
        response, _ = self.validity_pipeline.run(schema)
        choice = response.get(
            "Pipeline_root.block.layer[0].unit[CategoricalJudge Judge]_choice", None
        )
        return choice == "yes"


if __name__ == "__main__":
    evaluator = OptimizerEvaluator()

    optimizer_code = """
import torch

# Define parameter (requires_grad=True)
x = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(20):
    optimizer.zero_grad()
    loss = (x - 3) ** 2
    loss.backward()
    optimizer.step()
    print(f"Step {step + 1}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"\nOptimal x: {x.item():.4f}")
    """

    stdout = """
    Step 1: x = 0.0000, loss = 9.0000
    Step 2: x = 0.0900, loss = 8.1000
    Step 3: x = 0.1620, loss = 7.2900
    Step 4: x = 0.2187, loss = 6.5610
    Step 5: x = 0.2624, loss = 5.9049
    Step 6: x = 0.2962, loss = 5.3144
    Step 7: x = 0.3225, loss = 4.7830
    """

    stderr = """
    Traceback (most recent call last):
    """

    score = evaluator.check_validity(
        optimizer_code=optimizer_code, stdout=stdout, stderr=stderr
    )
    print(score)
