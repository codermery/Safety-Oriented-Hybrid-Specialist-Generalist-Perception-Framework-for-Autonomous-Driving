import json
import os

json_path = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/corner_case.json"
images_dir = "/workspace/siu_revision/data/coda_dataset/CODA/base-val-1500/images"
with open(json_path, 'r') as f:
    data = json.load(f)

missing = 0
found = 0
for img in data["images"]:
    fn = img["file_name"]
    if fn.startswith("images/"):
        fn = fn.replace("images/", "")
    full_path = os.path.join(images_dir, fn)
    if os.path.exists(full_path):
        found += 1
    else:
        missing += 1
        if missing <= 10:
            print(f"Missing: {fn} (tried path: {full_path})")

print(f"Total found: {found}, Total missing: {missing}")
