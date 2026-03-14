#!/usr/bin/env python3
"""Fix duplicate keys in Intel AutoRound safetensors shards.

Intel/Qwen3.5-397B-A17B-int4-AutoRound has 2688 duplicate weight keys
across shard 39 and 40. This script removes duplicates from the shard
that is NOT referenced by model.safetensors.index.json.

Usage:
    python3 fix_duplicate_keys.py /path/to/model
    python3 fix_duplicate_keys.py /path/to/model --verify-only
"""

import os
import sys
import json
import collections

def find_duplicates(model_dir):
    from safetensors import safe_open

    files = sorted(f for f in os.listdir(model_dir)
                   if f.endswith('.safetensors') and not f.endswith('.bak'))

    key_to_file = {}
    dupes = []
    for fname in files:
        path = os.path.join(model_dir, fname)
        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                if key in key_to_file:
                    dupes.append((key, key_to_file[key], fname))
                else:
                    key_to_file[key] = fname

    return dupes, len(key_to_file)

def fix_duplicates(model_dir):
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        idx = json.load(f)

    # Find which files have duplicates
    dupes, unique_count = find_duplicates(model_dir)
    if not dupes:
        print(f"No duplicates found ({unique_count} unique keys). Nothing to fix.")
        return

    # Group by file pairs
    file_pairs = collections.defaultdict(list)
    for key, f1, f2 in dupes:
        file_pairs[(f1, f2)].append(key)

    print(f"Found {len(dupes)} duplicate keys across {len(file_pairs)} file pair(s):")
    for (f1, f2), keys in file_pairs.items():
        print(f"  {f1} <-> {f2}: {len(keys)} duplicates")

    # For each duplicate, check which file the index points to (= keep that one)
    for (f1, f2), keys in file_pairs.items():
        # Determine which file to clean
        in_f1 = sum(1 for k in keys if idx['weight_map'].get(k) == f1)
        in_f2 = sum(1 for k in keys if idx['weight_map'].get(k) == f2)
        print(f"  Index says: {in_f1} in {f1}, {in_f2} in {f2}")

        # Remove dupes from the file NOT referenced by index
        if in_f2 >= in_f1:
            remove_from = f1
            keep_keys_from = f2
        else:
            remove_from = f2
            keep_keys_from = f1

        remove_path = os.path.join(model_dir, remove_from)
        keep_path = os.path.join(model_dir, keep_keys_from)

        # Get keys to remove (those that exist in the keep file)
        with safe_open(keep_path, framework='pt') as f:
            keep_file_keys = set(f.keys())

        # Load the file to clean, keeping only non-duplicate keys
        tensors = {}
        with safe_open(remove_path, framework='pt') as f:
            total = len(list(f.keys()))
            for key in f.keys():
                if key not in keep_file_keys:
                    tensors[key] = f.get_tensor(key)

        removed = total - len(tensors)
        print(f"\n  Fixing {remove_from}: {total} -> {len(tensors)} keys ({removed} removed)")

        # Backup and save
        backup = remove_path + '.bak'
        if not os.path.exists(backup):
            os.rename(remove_path, backup)
            print(f"  Backup: {backup}")
        else:
            os.remove(remove_path)
            print(f"  Backup already exists, overwriting shard")

        save_file(tensors, remove_path)
        size_gb = os.path.getsize(remove_path) / 1e9
        print(f"  Saved: {remove_from} ({size_gb:.2f} GB)")

    # Verify
    dupes_after, unique_after = find_duplicates(model_dir)
    print(f"\nVerification: {unique_after} unique keys, {len(dupes_after)} duplicates")
    if dupes_after:
        print("WARNING: Still has duplicates!")
        return False
    print("OK — all duplicates removed.")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [--verify-only]")
        sys.exit(1)

    model_dir = sys.argv[1]
    verify_only = '--verify-only' in sys.argv

    if verify_only:
        dupes, unique = find_duplicates(model_dir)
        print(f"{unique} unique keys, {len(dupes)} duplicates")
        if dupes:
            for k, f1, f2 in dupes[:5]:
                print(f"  {k}: {f1} <-> {f2}")
            if len(dupes) > 5:
                print(f"  ... and {len(dupes)-5} more")
        sys.exit(1 if dupes else 0)
    else:
        success = fix_duplicates(model_dir)
        sys.exit(0 if success else 1)
