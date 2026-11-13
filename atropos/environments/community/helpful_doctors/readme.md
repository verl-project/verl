Persona-Aware MedQA Benchmarking
https://youtube.com/shorts/02GEURik0PQ

Wandb: https://wandb.ai/nous-hackathon-2/atropos-environments_hack0_doctor_agent?nw=nwusertsadpbb
We intended on adding a simple percentage accurate score but couldn't get it done in time :(

In this project, we reimagined medical QA evaluation by introducing a persona filter—a novel layer that simulates real-world variability in patient communication styles. Leveraging the MedQA dataset as our foundation, we infused each scenario with distinct personas generated via xAI’s language models:

1. The Cooperative Patient – open, verbose, and highly informative.
2. The Reluctant Patient – terse, vague, and occasionally evasive.
3. The Neutral Patient – brief but factually consistent.

The clinical challenge we explored is simple but critical: Can a medical reasoning system consistently arrive at the correct diagnosis or treatment recommendation regardless of how the patient presents information?

Our pipeline works as follows:

Each original MedQA item (stem + multiple choice answers) is enriched with a synthetic patient interaction that simulates one of the three personas.
We maintain the original clinical question and choices.
Only the narrative context—the patient's communication—changes, testing robustness against dialogue variability.
This mirrors how real doctors must interpret patient symptoms, which are often incomplete or colored by personality, emotion, or context.

Why this matters:
Most QA benchmarks assume a perfect narrator. But in the real world, AI systems in healthcare will need to make decisions with varying degrees of input clarity.

Our approach stress-tests reasoning models under more human-like variability, offering a path toward safer and more empathetic medical AI.

Future Potential:

Extendable to reinforcement learning pipelines where the agent adapts its questioning strategy based on persona.
Can be used to benchmark bedside AI assistants, triage bots, or LLMs deployed in low-resource clinics.
Encourages development of models that ask better follow-up questions, not just give answers.
By combining structured medical QA with naturalistic persona variation, our project brings a crucial human dimension to the next generation of AI-health interfaces.
