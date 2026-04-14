from pathlib import Path

from release_benchmark.methods.registry import TRAIN_METHODS, ZEROSHOT_METHODS


def _module_to_file(module_name: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    rel = module_name.split(".")[1:]
    return root.joinpath(*rel).with_suffix(".py")


def run_registry_audit():
    rows = []
    for scope, table in (("train", TRAIN_METHODS), ("zeroshot", ZEROSHOT_METHODS)):
        for method, (module_name, class_name) in table.items():
            f = _module_to_file(module_name)
            if not f.exists():
                rows.append((scope, method, module_name, class_name, "missing file"))
                continue
            txt = f.read_text(encoding="utf-8", errors="ignore")
            ok = (f"class {class_name}(" in txt) or (f"class {class_name}:" in txt)
            rows.append(
                (
                    scope,
                    method,
                    module_name,
                    class_name,
                    "ok" if ok else "class missing",
                )
            )
    return rows
