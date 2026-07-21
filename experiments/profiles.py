"""Dataset registry loading and processed-profile discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

from .schema import PreparedProfile, SENTIMENT_ORDER, SplitData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.json"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / ".experiment_cache" / "processed"


def load_registry(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if tuple(registry.get("sentiment_order", ())) != SENTIMENT_ORDER:
        raise ValueError(
            f"Registry sentiment_order must be {list(SENTIMENT_ORDER)}, "
            f"got {registry.get('sentiment_order')}"
        )
    profiles = registry.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Dataset registry has no profiles")
    return registry


def resolve_profile_ids(requested: Iterable[str], registry: Dict[str, Any]) -> List[str]:
    values = list(requested)
    if not values or values == ["all"] or "all" in values:
        return list(registry["profiles"].keys())
    unknown = sorted(set(values) - set(registry["profiles"]))
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}")
    return values


def load_prepared_profile(
    profile_id: str,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> PreparedProfile:
    registry = load_registry(config_path)
    config = registry["profiles"].get(profile_id)
    if config is None:
        raise ValueError(f"Unknown profile: {profile_id}")

    root = Path(cache_root).resolve() / profile_id
    metadata_path = root / "metadata.json"
    audit_path = root / "dataset_audit.json"
    if not metadata_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            f"Prepared profile not found at {root}. Run scripts/prepare_experiment_data.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    aspects = list(config.get("aspects", []))
    task = str(config["task"])

    splits = {
        name: SplitData.from_jsonl(
            root / f"{name}.jsonl",
            profile_id=profile_id,
            task=task,
            split=name,
            aspects=aspects,
        )
        for name in ("train", "dev", "test")
    }
    profile = PreparedProfile(
        profile_id=profile_id,
        task=task,
        aspects=aspects,
        train=splits["train"],
        dev=splits["dev"],
        test=splits["test"],
        audit=audit,
        metadata=metadata,
    )
    profile.validate()
    return profile
