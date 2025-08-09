#!/usr/bin/env python3
import json, io, os, re

IN_PATH  = "/vulcanscratch/mhoover4/code/instruction_following_eval/data/input_data_hf_orig.jsonl"
OUT_PATH = "/vulcanscratch/mhoover4/code/instruction_following_eval/data/input_data_hf.fixed.jsonl"

removed_nulls = 0
coerced_to_int = 0
total_rows = 0

def coerce_value(key, v):
    """Drop None; coerce integer-like floats/strings to int; leave everything else."""
    global coerced_to_int, removed_nulls
    if v is None:
        removed_nulls += 1
        return None  # caller will drop it

    # Coerce floats like 2.0 -> 2
    if isinstance(v, float) and v.is_integer():
        coerced_to_int += 1
        return int(v)

    # Coerce numeric strings like "3" or "4.0" -> 3/4
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            coerced_to_int += 1
            return int(s)
        try:
            f = float(s)
            if f.is_integer():
                coerced_to_int += 1
                return int(f)
        except ValueError:
            pass

    return v

with io.open(IN_PATH, "r", encoding="utf-8") as fin, io.open(OUT_PATH, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        total_rows += 1

        # Ensure kwargs aligns with instruction_id_list
        kw_list = obj.get("kwargs")
        if not isinstance(kw_list, list):
            kw_list = []
        id_list = obj.get("instruction_id_list") or []
        if len(kw_list) < len(id_list):
            kw_list = kw_list + [{} for _ in range(len(id_list) - len(kw_list))]
        elif len(kw_list) > len(id_list):
            kw_list = kw_list[:len(id_list)]

        fixed_kwargs = []
        for d in kw_list:
            if not isinstance(d, dict):
                fixed_kwargs.append({})
                continue
            filtered = {}
            for k, v in d.items():
                vv = coerce_value(k, v)
                if vv is not None:
                    filtered[k] = vv
            fixed_kwargs.append(filtered)

        obj["kwargs"] = fixed_kwargs
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote: {OUT_PATH}")
print(f"Processed {total_rows} rows; removed {removed_nulls} nulls; coerced {coerced_to_int} values to int.")