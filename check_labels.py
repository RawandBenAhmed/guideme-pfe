from pathlib import Path

CLASSES = ['anesthesie_locale','connexion_bocal','desinfection_betadine',
           'diagnostique','dissection_sous_cutanee','incision_cutanee',
           'insertion_drain','preparation_patient']

counts = {c: {"video1":0,"video2":0,"video3":0} for c in CLASSES}

for split in ['train','val','test']:
    lbl_dir = Path(f'dataset_yolo/labels/{split}')
    if not lbl_dir.exists():
        continue
    for txt in lbl_dir.glob('*.txt'):
        content = txt.read_text().strip()
        if not content:
            continue
        cls_id   = int(content.split()[0])
        cls_name = CLASSES[cls_id]
        name     = txt.stem  # ex: video1_frame_00001

        for v in ["video1","video2","video3"]:
            if v in name:
                counts[cls_name][v] += 1
                break

print(f"{'Etape':<30} {'video1':>8} {'video2':>8} {'video3':>8}")
print("-" * 58)
for cls, vids in sorted(counts.items()):
    print(f"{cls:<30} {vids['video1']:>8} {vids['video2']:>8} {vids['video3']:>8}")