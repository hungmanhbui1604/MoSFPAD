import glob
import json
import os
import random
from typing import Callable, Optional

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def _extract_id(path: str, id_type: str = "subject") -> str:
    assert id_type in ("subject", "finger"), (
        f"Invalid id type '{id_type}'. Choose from: ('subject', 'finger'])"
    )

    norm = path.replace("\\", "/")

    if "LivDet" in norm:
        """
        data/LivDet/LivDet2011/Biometrika/Train/Live/1_1.png
        data/LivDet/LivDet2013/Biometrika/Train/Live/031_Lidx_1.png
        data/LivDet/LivDet2009/CrossMatch/Train/Live/0056195_R1_3/0056195_R1_1.png
        """
        path_parts = norm.split("/")
        livdet_idx = path_parts.index("LivDet")
        year = path_parts[livdet_idx + 1]
        sensor = path_parts[livdet_idx + 2]

        filename = os.path.basename(norm)
        parts = filename.split("_")
        assert 3 >= len(parts) >= 2, f"Unexpected LivDet filename format: {filename}"

        if len(parts) == 3:  # subjectid_finger_impression
            subject_id, finger_id, _ = parts
        else:  # id_impression
            finger_id, _ = parts
            subject_id = finger_id

        dst = f"{year}_{sensor}"

    elif "ATVS-FF" in norm:
        # 03_li_1.bmp
        filename = os.path.basename(norm)
        parts = filename.split("_")
        assert len(parts) == 3, f"Unexpected ATVS-FF filename format: {filename}"

        subject_id = parts[0]
        finger_id = parts[1]

        dst = "atvsff"

    if id_type == "subject":
        return f"{dst}_{subject_id}"
    else:  # "finger"
        return f"{dst}_{subject_id}_{finger_id}"


def create_pad_splits(
    data_root: str,
    recog_output_path: str,
    pad_output_path: str,
    split_ratio: tuple = (0.6, 0.2, 0.2),
    min_samples: Optional[int] = 3,
    seed: int = 42,
) -> tuple[dict, dict]:
    def _filter_by_id(
        subject_finger_paths: dict[str, dict[str, list[str]]],
    ) -> dict[str, dict[str, list[str]]]:
        filtered_subject_finger_paths = {}
        removed_count = 0

        for subject, fingers in subject_finger_paths.items():
            valid_fingers = {}
            for finger, paths in fingers.items():
                count = sum(1 for p in paths if "Live" in p)
                if count >= min_samples:
                    valid_fingers[finger] = paths
                else:
                    removed_count += 1

            if valid_fingers:
                filtered_subject_finger_paths[subject] = valid_fingers

        print(f"Removed {removed_count} fingers that did not meet the criteria.")
        return filtered_subject_finger_paths

    assert len(split_ratio) == 3, "split_ratio must have 3 values (train, val, test)"
    assert all(r >= 0 for r in split_ratio), "split_ratio must be non-negative"
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"
    assert min_samples >= 2, "min_samples must be at least 2"

    print(f"Creating splits for {data_root}")

    exts = ("*.bmp", "*.tif", "*.png", "*.jpg")
    all_paths = [
        p
        for ext in exts
        for p in glob.glob(os.path.join(data_root, "**", ext), recursive=True)
    ]

    subject_finger_paths = {}
    for path in all_paths:
        subject = _extract_id(path, "subject")
        finger = _extract_id(path, "finger")
        if subject not in subject_finger_paths:
            subject_finger_paths[subject] = {}
        if finger not in subject_finger_paths[subject]:
            subject_finger_paths[subject][finger] = []
        subject_finger_paths[subject][finger].append(path)

    if min_samples is not None:
        recog_subject_finger_paths = _filter_by_id(subject_finger_paths)

    all_subjects = sorted(subject_finger_paths.keys())

    rng = random.Random(seed)
    rng.shuffle(all_subjects)

    n_total = len(all_subjects)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_subjects = set(all_subjects[:n_train])
    val_subjects = set(all_subjects[n_train : n_train + n_val])

    recog_splits = {
        "train": {},
        "val": {},
        "test": {},
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }

    pad_splits = {
        "train": {},
        "val": {},
        "test": {},
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }

    for subject, fingers in subject_finger_paths.items():
        target = (
            "train"
            if subject in train_subjects
            else "val"
            if subject in val_subjects
            else "test"
        )
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in pad_splits[target]:
                        pad_splits[target][finger] = []
                    pad_splits[target][finger].append((path, 0))
                    pad_splits[f"{target}_samples"] += 1
                else:
                    if finger not in pad_splits[target]:
                        pad_splits[target][finger] = []
                    pad_splits[target][finger].append((path, 1))
                    pad_splits[f"{target}_samples"] += 1

    for subject, fingers in recog_subject_finger_paths.items():
        target = (
            "train"
            if subject in train_subjects
            else "val"
            if subject in val_subjects
            else "test"
        )
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in recog_splits[target]:
                        recog_splits[target][finger] = []
                    recog_splits[target][finger].append(path)
                    recog_splits[f"{target}_samples"] += 1

    recog_train_subjects_cnt = len(
        set(s for s in recog_subject_finger_paths.keys() if s in train_subjects)
    )
    recog_val_subjects_cnt = len(
        set(s for s in recog_subject_finger_paths.keys() if s in val_subjects)
    )
    recog_test_subjects_cnt = len(
        set(
            s
            for s in recog_subject_finger_paths.keys()
            if s not in train_subjects and s not in val_subjects
        )
    )

    recog_splits.update(
        {
            "train_subjects": recog_train_subjects_cnt,
            "val_subjects": recog_val_subjects_cnt,
            "test_subjects": recog_test_subjects_cnt,
            "train_fingers": len(recog_splits["train"]),
            "val_fingers": len(recog_splits["val"]),
            "test_fingers": len(recog_splits["test"]),
            "total_subjects": recog_train_subjects_cnt
            + recog_val_subjects_cnt
            + recog_test_subjects_cnt,
            "total_fingers": len(recog_splits["train"])
            + len(recog_splits["val"])
            + len(recog_splits["test"]),
            "total_samples": recog_splits["train_samples"]
            + recog_splits["val_samples"]
            + recog_splits["test_samples"],
        }
    )

    pad_splits.update(
        {
            "train_subjects": n_train,
            "val_subjects": n_val,
            "test_subjects": n_total - n_train - n_val,
            "train_fingers": len(pad_splits["train"]),
            "val_fingers": len(pad_splits["val"]),
            "test_fingers": len(pad_splits["test"]),
            "total_subjects": n_total,
            "total_fingers": len(pad_splits["train"])
            + len(pad_splits["val"])
            + len(pad_splits["test"]),
            "total_samples": pad_splits["train_samples"]
            + pad_splits["val_samples"]
            + pad_splits["test_samples"],
        }
    )

    print(
        "Recog:\n"
        f"• Train: {recog_splits['train_samples']} samples ({recog_splits['train_subjects']} subjects / {recog_splits['train_fingers']} fingers)\n"
        f"• Val: {recog_splits['val_samples']} samples ({recog_splits['val_subjects']} subjects / {recog_splits['val_fingers']} fingers)\n"
        f"• Test: {recog_splits['test_samples']} samples ({recog_splits['test_subjects']} subjects / {recog_splits['test_fingers']} fingers)\n"
        f"Total: {recog_splits['total_samples']} samples ({recog_splits['total_subjects']} subjects / {recog_splits['total_fingers']} fingers)",
    )

    print(
        "PAD:\n"
        f"• Train: {pad_splits['train_samples']} samples ({pad_splits['train_subjects']} subjects / {pad_splits['train_fingers']} fingers)\n"
        f"• Val: {pad_splits['val_samples']} samples ({pad_splits['val_subjects']} subjects / {pad_splits['val_fingers']} fingers)\n"
        f"• Test: {pad_splits['test_samples']} samples ({pad_splits['test_subjects']} subjects / {pad_splits['test_fingers']} fingers)\n"
        f"Total: {pad_splits['total_samples']} samples ({pad_splits['total_subjects']} subjects / {pad_splits['total_fingers']} fingers)",
    )

    os.makedirs(os.path.dirname(recog_output_path), exist_ok=True)
    with open(recog_output_path, "w") as f:
        json.dump(recog_splits, f, indent=2)

    os.makedirs(os.path.dirname(pad_output_path), exist_ok=True)
    with open(pad_output_path, "w") as f:
        json.dump(pad_splits, f, indent=2)

    return recog_splits, pad_splits


def create_LivDet_splits(
    data_root: str = "data/LivDet",
    recog_output_path: str = "data/LivDet/recog_splits.json",
    pad_output_path: str = "data/LivDet/pad_splits.json",
    val_ratio: float = 0.2,
    min_samples: int = 3,
    seed: int = 42,
) -> tuple[dict, dict]:
    def _filter_by_id(
        subject_finger_paths: dict[str, dict[str, list[str]]],
    ) -> dict[str, dict[str, list[str]]]:
        filtered_subject_finger_paths = {}
        removed_count = 0

        for subject, fingers in subject_finger_paths.items():
            valid_fingers = {}
            for finger, paths in fingers.items():
                count = sum(1 for p in paths if "Live" in p)
                if count >= min_samples:
                    valid_fingers[finger] = paths
                else:
                    removed_count += 1

            if valid_fingers:
                filtered_subject_finger_paths[subject] = valid_fingers

        print(f"Removed {removed_count} fingers that did not meet the criteria.")
        return filtered_subject_finger_paths

    assert 0.0 <= val_ratio <= 1.0, "val_ratio must be between 0 and 1"
    assert min_samples >= 2, "min_samples must be at least 2"

    print(f"Creating splits for {data_root}")

    exts = ("*.bmp", "*.tif", "*.png", "*.jpg")
    all_paths = [
        p
        for ext in exts
        for p in glob.glob(os.path.join(data_root, "**", ext), recursive=True)
    ]

    train_subject_finger_paths = {}
    test_subject_finger_paths = {}
    for path in all_paths:
        if "Train" in path:
            subject = _extract_id(path, "subject")
            finger = _extract_id(path, "finger")
            if subject not in train_subject_finger_paths:
                train_subject_finger_paths[subject] = {}
            if finger not in train_subject_finger_paths[subject]:
                train_subject_finger_paths[subject][finger] = []
            train_subject_finger_paths[subject][finger].append(path)
        elif "Test" in path:
            subject = _extract_id(path, "subject")
            finger = _extract_id(path, "finger")
            if subject not in test_subject_finger_paths:
                test_subject_finger_paths[subject] = {}
            if finger not in test_subject_finger_paths[subject]:
                test_subject_finger_paths[subject][finger] = []
            test_subject_finger_paths[subject][finger].append(path)

    if min_samples is not None:
        recog_train_subject_finger_paths = _filter_by_id(train_subject_finger_paths)
        recog_test_subject_finger_paths = _filter_by_id(test_subject_finger_paths)

    all_train_subjects = sorted(train_subject_finger_paths.keys())

    rng = random.Random(seed)
    rng.shuffle(all_train_subjects)

    n_total = len(all_train_subjects)
    n_train = int(n_total * (1 - val_ratio))

    train_subjects = set(all_train_subjects[:n_train])

    recog_splits = {
        "train": {},
        "val": {},
        "test": {},
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }

    pad_splits = {
        "train": {},
        "val": {},
        "test": {},
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }

    for subject, fingers in train_subject_finger_paths.items():
        target = "train" if subject in train_subjects else "val"
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in pad_splits[target]:
                        pad_splits[target][finger] = []
                    pad_splits[target][finger].append((path, 0))
                    pad_splits[f"{target}_samples"] += 1
                else:
                    if finger not in pad_splits[target]:
                        pad_splits[target][finger] = []
                    pad_splits[target][finger].append((path, 1))
                    pad_splits[f"{target}_samples"] += 1

    for subject, fingers in recog_train_subject_finger_paths.items():
        target = "train" if subject in train_subjects else "val"
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in recog_splits[target]:
                        recog_splits[target][finger] = []
                    recog_splits[target][finger].append(path)
                    recog_splits[f"{target}_samples"] += 1

    for subject, fingers in test_subject_finger_paths.items():
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in pad_splits["test"]:
                        pad_splits["test"][finger] = []
                    pad_splits["test"][finger].append((path, 0))
                    pad_splits["test_samples"] += 1
                else:
                    if finger not in pad_splits["test"]:
                        pad_splits["test"][finger] = []
                    pad_splits["test"][finger].append((path, 1))
                    pad_splits["test_samples"] += 1

    for subject, fingers in recog_test_subject_finger_paths.items():
        for finger, paths in fingers.items():
            for path in paths:
                if "Live" in path:
                    if finger not in recog_splits["test"]:
                        recog_splits["test"][finger] = []
                    recog_splits["test"][finger].append(path)
                    recog_splits["test_samples"] += 1

    recog_train_subjects_cnt = len(
        set(s for s in recog_train_subject_finger_paths.keys() if s in train_subjects)
    )
    recog_val_subjects_cnt = len(
        set(
            s
            for s in recog_train_subject_finger_paths.keys()
            if s not in train_subjects
        )
    )
    recog_test_subjects_cnt = len(recog_test_subject_finger_paths)

    recog_splits.update(
        {
            "train_subjects": recog_train_subjects_cnt,
            "val_subjects": recog_val_subjects_cnt,
            "test_subjects": recog_test_subjects_cnt,
            "train_fingers": len(recog_splits["train"]),
            "val_fingers": len(recog_splits["val"]),
            "test_fingers": len(recog_splits["test"]),
            "total_subjects": recog_train_subjects_cnt
            + recog_val_subjects_cnt
            + recog_test_subjects_cnt,
            "total_fingers": len(recog_splits["train"])
            + len(recog_splits["val"])
            + len(recog_splits["test"]),
            "total_samples": recog_splits["train_samples"]
            + recog_splits["val_samples"]
            + recog_splits["test_samples"],
        }
    )

    pad_splits.update(
        {
            "train_subjects": n_train,
            "val_subjects": n_total - n_train,
            "test_subjects": len(test_subject_finger_paths),
            "train_fingers": len(pad_splits["train"]),
            "val_fingers": len(pad_splits["val"]),
            "test_fingers": len(pad_splits["test"]),
            "total_subjects": n_total + len(test_subject_finger_paths),
            "total_fingers": len(pad_splits["train"])
            + len(pad_splits["val"])
            + len(pad_splits["test"]),
            "total_samples": pad_splits["train_samples"]
            + pad_splits["val_samples"]
            + pad_splits["test_samples"],
        }
    )

    print(
        "Recog:\n"
        f"• Train: {recog_splits['train_samples']} samples ({recog_splits['train_subjects']} subjects / {recog_splits['train_fingers']} fingers)\n"
        f"• Val: {recog_splits['val_samples']} samples ({recog_splits['val_subjects']} subjects / {recog_splits['val_fingers']} fingers)\n"
        f"• Test: {recog_splits['test_samples']} samples ({recog_splits['test_subjects']} subjects / {recog_splits['test_fingers']} fingers)\n"
        f"Total: {recog_splits['total_samples']} samples ({recog_splits['total_subjects']} subjects / {recog_splits['total_fingers']} fingers)",
    )

    print(
        "PAD:\n"
        f"• Train: {pad_splits['train_samples']} samples ({pad_splits['train_subjects']} subjects / {pad_splits['train_fingers']} fingers)\n"
        f"• Val: {pad_splits['val_samples']} samples ({pad_splits['val_subjects']} subjects / {pad_splits['val_fingers']} fingers)\n"
        f"• Test: {pad_splits['test_samples']} samples ({pad_splits['test_subjects']} subjects / {pad_splits['test_fingers']} fingers)\n"
        f"Total: {pad_splits['total_samples']} samples ({pad_splits['total_subjects']} subjects / {pad_splits['total_fingers']} fingers)",
    )

    os.makedirs(os.path.dirname(recog_output_path), exist_ok=True)
    with open(recog_output_path, "w") as f:
        json.dump(recog_splits, f, indent=2)

    os.makedirs(os.path.dirname(pad_output_path), exist_ok=True)
    with open(pad_output_path, "w") as f:
        json.dump(pad_splits, f, indent=2)

    return recog_splits, pad_splits


def unify_pad_splits(
    recog_split_paths: Optional[list[str]] = None,
    pad_split_paths: Optional[list[str]] = None,
    output_path: str = "data/pad_splits.json",
):
    recog_split_paths = recog_split_paths or []
    pad_split_paths = pad_split_paths or []

    print(f"Unifying PAD splits from {recog_split_paths} and {pad_split_paths}")

    unified = {
        "train": {},
        "val": {},
        "test": {},
        "train_subjects": 0,
        "val_subjects": 0,
        "test_subjects": 0,
        "train_fingers": 0,
        "val_fingers": 0,
        "test_fingers": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "total_subjects": 0,
        "total_fingers": 0,
        "total_samples": 0,
    }

    for split_path in recog_split_paths:
        with open(split_path, "r") as f:
            splits = json.load(f)

        for split in ("train", "val", "test"):
            transformed = {k: [[p, 0] for p in v] for k, v in splits[split].items()}
            unified[split].update(transformed)

        unified["train_subjects"] += splits["train_subjects"]
        unified["val_subjects"] += splits["val_subjects"]
        unified["test_subjects"] += splits["test_subjects"]
        unified["train_fingers"] += splits["train_fingers"]
        unified["val_fingers"] += splits["val_fingers"]
        unified["test_fingers"] += splits["test_fingers"]
        unified["train_samples"] += splits["train_samples"]
        unified["val_samples"] += splits["val_samples"]
        unified["test_samples"] += splits["test_samples"]

    for split_path in pad_split_paths:
        with open(split_path, "r") as f:
            splits = json.load(f)

        unified["train"].update(splits["train"])
        unified["val"].update(splits["val"])
        unified["test"].update(splits["test"])

        unified["train_subjects"] += splits["train_subjects"]
        unified["val_subjects"] += splits["val_subjects"]
        unified["test_subjects"] += splits["test_subjects"]
        unified["train_fingers"] += splits["train_fingers"]
        unified["val_fingers"] += splits["val_fingers"]
        unified["test_fingers"] += splits["test_fingers"]
        unified["train_samples"] += splits["train_samples"]
        unified["val_samples"] += splits["val_samples"]
        unified["test_samples"] += splits["test_samples"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    unified["total_subjects"] = (
        unified["train_subjects"] + unified["val_subjects"] + unified["test_subjects"]
    )
    unified["total_fingers"] = (
        unified["train_fingers"] + unified["val_fingers"] + unified["test_fingers"]
    )
    unified["total_samples"] = (
        unified["train_samples"] + unified["val_samples"] + unified["test_samples"]
    )

    print(
        f"• Train: {unified['train_samples']} samples ({unified['train_subjects']} subjects / {unified['train_fingers']} fingers)\n"
        f"• Val: {unified['val_samples']} samples ({unified['val_subjects']} subjects / {unified['val_fingers']} fingers)\n"
        f"• Test: {unified['test_samples']} samples ({unified['test_subjects']} subjects / {unified['test_fingers']} fingers)\n"
        f"Total: {unified['total_samples']} samples ({unified['total_subjects']} subjects / {unified['total_fingers']} fingers)"
    )

    return unified


class PADDataset(Dataset):
    def __init__(
        self,
        split_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.transform = transform

        with open(split_path) as f:
            finger_to_paths = json.load(f)[split]

        self.samples = []
        for finger, paths in finger_to_paths.items():
            self.samples.extend(paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        n_live = sum(1 for (_, label) in self.samples if label == 0)
        n_spoof = len(self) - n_live
        return f"PADDataset: {len(self)} samples (live: {n_live}, spoof: {n_spoof})"


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    livdet_data_roots = [
        "data/LivDet/LivDet2009/Biometrika",
        "data/LivDet/LivDet2009/CrossMatch",
        "data/LivDet/LivDet2009/Identix",
        "data/LivDet/LivDet2011/Biometrika",
        "data/LivDet/LivDet2011/Digital",
        "data/LivDet/LivDet2011/Italdata",
        "data/LivDet/LivDet2011/Sagem",
        "data/LivDet/LivDet2013/Biometrika",
        "data/LivDet/LivDet2013/CrossMatch",
        "data/LivDet/LivDet2013/Italdata",
        "data/LivDet/LivDet2015/Biometrika",
        "data/LivDet/LivDet2015/CrossMatch",
        "data/LivDet/LivDet2015/DigitalPersona",
        "data/LivDet/LivDet2015/GreenBit",
        "data/LivDet/LivDet2015/HiScan",
    ]
    for root in livdet_data_roots:
        create_LivDet_splits(
            data_root=root,
            recog_output_path=f"{root}/recog_splits.json",
            pad_output_path=f"{root}/pad_splits.json",
            val_ratio=0.3,
            min_samples=3,
            seed=42,
        )
        print()

    create_pad_splits(
        data_root="data/ATVS-FF",
        recog_output_path="data/ATVS-FF/recog_splits.json",
        pad_output_path="data/ATVS-FF/pad_splits.json",
        split_ratio=(1, 0, 0),
        min_samples=3,
        seed=42,
    )
    print()

    unify_pad_splits(
        pad_split_paths=[
            "data/LivDet/LivDet2009/Biometrika/pad_splits.json",
            "data/LivDet/LivDet2009/CrossMatch/pad_splits.json",
            "data/LivDet/LivDet2009/Identix/pad_splits.json",
            "data/LivDet/LivDet2011/Biometrika/pad_splits.json",
            "data/LivDet/LivDet2011/Digital/pad_splits.json",
            "data/LivDet/LivDet2011/Italdata/pad_splits.json",
            "data/LivDet/LivDet2011/Sagem/pad_splits.json",
            "data/LivDet/LivDet2013/Biometrika/pad_splits.json",
            "data/LivDet/LivDet2013/CrossMatch/pad_splits.json",
            "data/LivDet/LivDet2013/Italdata/pad_splits.json",
            "data/LivDet/LivDet2015/Biometrika/pad_splits.json",
            "data/LivDet/LivDet2015/CrossMatch/pad_splits.json",
            "data/LivDet/LivDet2015/DigitalPersona/pad_splits.json",
            "data/LivDet/LivDet2015/GreenBit/pad_splits.json",
            "data/LivDet/LivDet2015/HiScan/pad_splits.json",
            "data/ATVS-FF/pad_splits.json",
        ],
        output_path="data/pad_splits.json",
    )
    print()

    pad_train_dataset = PADDataset(
        split_path="data/pad_splits.json",
        split="train",
        transform=transform,
    )
    print(pad_train_dataset)
    print()

    pad_val_dataset = PADDataset(
        split_path="data/pad_splits.json",
        split="val",
        transform=transform,
    )
    print(pad_val_dataset)
    print()

    pad_test_dataset = PADDataset(
        split_path="data/pad_splits.json",
        split="test",
        transform=transform,
    )
    print(pad_test_dataset)
    print()