#!/usr/bin/env bash
set -euox pipefail


# Define config specifications: "config_name:output_file:config_arg"
CONFIG_SPECS=(
    "ppo_trainer:_generated_ppo_trainer.yaml:"
    "ppo_megatron_trainer:_generated_ppo_megatron_trainer.yaml:--config-name=ppo_megatron_trainer.yaml"
    "ppo_diffusion_trainer:_generated_ppo_diffusion_trainer.yaml:--config-name=ppo_diffusion_trainer.yaml"
    "ppo_trainer:_generated_ppo_veomni_trainer.yaml:model_engine=veomni"
)

generate_config() {
    local config_name="$1"
    local output_file="$2"
    local config_arg="$3"
    
    local target_cfg="verl/trainer/config/${output_file}"
    local tmp_header=$(mktemp)
    local tmp_cfg=$(mktemp)
    
    echo "# This reference configration yaml is automatically generated via 'scripts/generate_trainer_config.sh'" > "$tmp_header"
    echo "# in which it invokes 'python3 scripts/print_cfg.py --cfg job ${config_arg}' to flatten the 'verl/trainer/config/${config_name}.yaml' config fields into a single file." >> "$tmp_header"
    echo "# Do not modify this file directly." >> "$tmp_header"
    echo "# The file is usually only for reference and never used." >> "$tmp_header"
    echo "" >> "$tmp_header"
    
    python3 scripts/print_cfg.py --cfg job ${config_arg} > "$tmp_cfg"
    
    cat "$tmp_header" > "$target_cfg"
    sed -n '/^actor_rollout_ref/,$p' "$tmp_cfg" >> "$target_cfg"
    
    rm "$tmp_cfg" "$tmp_header"
    
    echo "Generated: $target_cfg"
}

for spec in "${CONFIG_SPECS[@]}"; do
    IFS=':' read -r config_name output_file config_arg <<< "$spec"
    generate_config "$config_name" "$output_file" "$config_arg"
done

for spec in "${CONFIG_SPECS[@]}"; do
    IFS=':' read -r config_name output_file config_arg <<< "$spec"
    target_cfg="verl/trainer/config/${output_file}"
    if ! git diff --exit-code -- "$target_cfg" >/dev/null; then
        echo "✖ $target_cfg is out of date. Please regenerate via 'scripts/generate_trainer_config.sh' and commit the changes."
        exit 1
    fi
done

echo "All good"
exit 0
