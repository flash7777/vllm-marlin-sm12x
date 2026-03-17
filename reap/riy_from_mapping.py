#!/usr/bin/env python3
"""Generate RIY profile from expert_mapping_pruned.json."""
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned", required=True, help="expert_mapping_pruned.json")
    parser.add_argument("--model", default="Qwen3.5-397B-A17B")
    parser.add_argument("--workload", default="general 35pct (REAP-262B equivalent)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.pruned) as f:
        pruned = json.load(f)

    pruned_experts = []
    for layer_str, expert_ids in pruned.items():
        for eid in expert_ids:
            pruned_experts.append([int(layer_str), eid])

    profile = {
        "version": 1,
        "model": args.model,
        "quantization": "int4-autoround",
        "workload": args.workload,
        "pruned_experts": pruned_experts,
    }

    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"RIY profile: {len(pruned_experts)} pruned experts across {len(pruned)} layers")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
