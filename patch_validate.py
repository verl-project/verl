import pathlib

p = pathlib.Path('/cbs/cua/verl-async/verl/trainer/ppo/ray_trainer.py')
content = p.read_text()

old = '            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")\n\n            # pad to be divisible by dp_size'

new = '''            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # Debug: check if raw_prompt contains image info for multi-modal validation
            if "raw_prompt" in test_gen_batch.non_tensor_batch:
                sample_prompt = test_gen_batch.non_tensor_batch["raw_prompt"][0]
                has_image = any(
                    isinstance(c, dict) and c.get("type") == "image"
                    for msg in (sample_prompt if isinstance(sample_prompt, list) else [sample_prompt])
                    if isinstance(msg, dict)
                    for c in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
                )
                print(
                    f"[Validate Debug] raw_prompt found, has_image={has_image}, "
                    f"num_samples={len(test_gen_batch.non_tensor_batch['raw_prompt'])}, "
                    f"non_tensor_keys={list(test_gen_batch.non_tensor_batch.keys())}"
                )
            else:
                print("[Validate Debug] WARNING: raw_prompt NOT found in test_gen_batch.non_tensor_batch!")

            # pad to be divisible by dp_size'''

if old in content:
    content = content.replace(old, new, 1)
    p.write_text(content)
    print('SUCCESS: validate debug log added')
else:
    print('ERROR: old string not found')
