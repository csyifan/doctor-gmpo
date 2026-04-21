"""Convert KAMED session JSON files to JSONL format for VeRL RLHFDataset."""
import json
import os
import glob
import random

KAMED_BASE = "/nfs-stor/yifan.lu/dataset/KAMED/VRBot-sigir2021-datasets"
OUTPUT_DIR = "/home/yifan.lu/verl_doctor/traindata"

SYSTEM_PROMPT = "你是一位专业的医生助手。请根据患者的描述，进行问诊并给出诊断建议。"

def convert_session(session_path):
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = data.get("dialogues", [])
    if not dialogues or dialogues[0]["role"] != "patient":
        return None

    first_patient_msg = dialogues[0]["sentence"]
    disease = data.get("topic", "")
    disease_grad = data.get("disease_grad", "")

    # Build ground truth from doctor turns
    ground_truth = ""
    for d in dialogues:
        if d["role"] == "doctor":
            ground_truth = d["sentence"]  # last doctor response as ground truth
            break

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_patient_msg},
    ]

    entry = {
        "data_source": "kamed",
        "prompt": prompt,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "interaction_kwargs": {
                "name": os.path.basename(session_path),
                "query": first_patient_msg,
                "ground_truth": ground_truth,
                "disease": disease,
                "disease_grad": disease_grad,
            }
        },
    }
    return entry

def convert_split(split_name, max_samples=-1):
    split_dir = os.path.join(KAMED_BASE, f"kamed_{split_name}", f"kamed_{split_name}")
    files = sorted(glob.glob(os.path.join(split_dir, "*.session.json")))

    entries = []
    for f in files:
        entry = convert_session(f)
        if entry:
            entries.append(entry)

    if max_samples > 0:
        entries = entries[:max_samples]

    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"{split_name}: {len(entries)} samples -> {output_path}")
    return entries

if __name__ == "__main__":
    train = convert_split("train")
    valid = convert_split("valid")
    test = convert_split("test")

    # Debug sets (5 samples each)
    random.seed(42)
    for name, data in [("debug_train", train), ("debug_valid", valid)]:
        samples = random.sample(data, min(5, len(data)))
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{name}: {len(samples)} samples -> {path}")
