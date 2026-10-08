from src.data import load_split


def test_split_exact_deterministic_partition():
    import random
    ids = list(range(7473))
    random.Random(42).shuffle(ids)
    split = load_split()
    assert split == {"dev": sorted(ids[:500]), "train": sorted(ids[500:])}
    assert not set(split["train"]) & set(split["dev"])
