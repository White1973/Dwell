"""Fail-closed compatibility lock for the few components Le-WM still shares."""
from __future__ import annotations

import hashlib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]

# Updating an entry is an explicit compatibility event: run both ordinary-WM
# and Le-WM suites, inspect checkpoint semantics, then update the digest in the
# same reviewed change. Ordinary-WM work must never update this automatically.
LOCKED_SHARED_CORE_V1 = {
    "sembelief_wm/config.py": "a2f2644ac1b829b8661dd101cc1c99ab0ae527a66c385ff17bc5414c13f557bc",
    "sembelief_wm/data/datasource.py": "a94b7714b0298f4e4940816c883e557f027fffdfc5cd0aff77f578e41434ae18",
    "sembelief_wm/data/tokenizers/image.py": "bad017596621c952d7e6f477026f6ba8b24d6df9e2b5cd5649de49b9775b9b7e",
    "sembelief_wm/data/adapters/sokoban.py": "dc3a19e121e1f8405152c5ad305c868784c78af988aa36e1f2f35c6ca0483d0f",
    "sembelief_wm/envs/sokoban/action_adapter.py": "dd13621e809def2892e1b25297be3e502da625b56da4f1a91ad261091a912dac",
    "sembelief_wm/model/__init__.py": "1b9f7a94021bf99d3ad7904d3de852300c19c50853c1f47f3dbf804a487d08e7",
    "sembelief_wm/model/backbone_qwen.py": "4ca84ebfb46d72c6033c8b0e56e4348113a197dc92d01d181369428dcee337f8",
    "sembelief_wm/model/belief.py": "ee7449a8ff89d37e1f96421807087d154e9e1b8e96bafa9c3dcc90aba3a48d57",
    "sembelief_wm/model/checkpoint_semantics.py": "6fb418e861cca974b6e0fe65d1999ec13df59ee9642891e57d89eff1a2cad1d4",
    "sembelief_wm/model/policy_backbone.py": "dce11d18753905e6d078c0fa29f4ebb7a6f35abaee36e182108655ee372e6229",
    "sembelief_wm/model/reward.py": "6003c3f124f14beec6d5fb09607f4ee1432540d19c12fe0ef348b0c5c8bd3972",
    "sembelief_wm/model/world_model.py": "97e29a1601ab2d6c9785f0cea82fb7d90f802f381e0c8fd2e76b062172124a7d",
    "sembelief_wm/model/transition.py": "92ceb90d5f8d74d6d57b07892f8f5545313e082af34b297f10f31c9426d327b3",
    "sembelief_wm/types.py": "d19d476a9dc541a2cea78f5ec7ed0f5bbc2615c10c96d67082ff7d6f673f8fd4",
    "sembelief_wm/collectors/real.py": "f2e44ca9a1ca53be94f52d0b528968ff30db896e6b29ca13c91031bc82eb350f",
    "sembelief_wm/rl/llm_policy.py": "89bdd2179c9a1ff3889f950d09d461674e49b1d76f63dde2dacb22a98e6393ab",
    "sembelief_wm/rl/ppo.py": "a45869b9be9cb9dd791b8186f42503a1951f5f76055bc9686f0568f57c5a12fc",
    "sembelief_wm/pipelines/assemble.py": "0bb7f4cf02e5fedcebe16380e68262048c0977ff4485190a6a54dcf1b641a35b",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_shared_core() -> dict[str, str]:
    changed: dict[str, str] = {}
    for relative, expected in LOCKED_SHARED_CORE_V1.items():
        path = _ROOT / relative
        actual = "missing" if not path.is_file() else _sha256(path)
        if actual != expected:
            changed[relative] = f"expected={expected}, actual={actual}"
    if changed:
        details = "\n".join(
            f"  {name}: {value}" for name, value in changed.items()
        )
        raise RuntimeError(
            "Le-WM locked shared-core contract changed. Ordinary-WM edits "
            "cannot silently enter a Le-WM run. Run both compatibility suites "
            "and explicitly revise LOCKED_SHARED_CORE_V1 before proceeding:\n"
            + details
        )
    return dict(LOCKED_SHARED_CORE_V1)


if __name__ == "__main__":
    locked = verify_locked_shared_core()
    print(f"Le-WM shared-core isolation contract passed: {len(locked)} files locked")
