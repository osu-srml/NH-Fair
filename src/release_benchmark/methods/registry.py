"""Explicit method registry for release_benchmark."""

from importlib import import_module

TRAIN_METHODS = {
    # CV methods
    "erm": ("release_benchmark.methods.cv.erm", "erm"),
    "randaug": ("release_benchmark.methods.cv.randaug", "randaug"),
    "mixup": ("release_benchmark.methods.cv.mixup", "mixup"),
    "bm": ("release_benchmark.methods.cv.bm", "bm"),
    "decoupled": ("release_benchmark.methods.cv.decoupled", "decoupled"),
    "dfr": ("release_benchmark.methods.cv.dfr", "dfr"),
    "gapreg": ("release_benchmark.methods.cv.gapreg", "gapreg"),
    "mcdp": ("release_benchmark.methods.cv.mcdp", "mcdp"),
    "fis": ("release_benchmark.methods.cv.fis", "fis"),
    "fscl": ("release_benchmark.methods.cv.fscl", "fscl"),
    "groupdro": ("release_benchmark.methods.cv.groupdro", "groupdro"),
    "laftr": ("release_benchmark.methods.cv.laftr", "laftr"),
    "oxonfair_method": (
        "release_benchmark.methods.cv.oxonfair_method",
        "oxonfair_method",
    ),
    # VLM methods (train/finetune; base CLIP is zeroshot-only — see ZEROSHOT_METHODS)
    "clip_sfid": ("release_benchmark.methods.vlm.clip_sfid", "clip_sfid"),
    "clip_fairer": ("release_benchmark.methods.vlm.clip_fairer", "clip_fairer"),
}

ZEROSHOT_METHODS = {
    "clip": ("release_benchmark.methods.vlm.clip", "clip"),
    "blip2": ("release_benchmark.methods.vlm.blip2", "blip2"),
    "qwen": ("release_benchmark.methods.lvlm.qwen", "qwen"),
    "llama": ("release_benchmark.methods.lvlm.llama", "llama"),
    "llama4": ("release_benchmark.methods.lvlm.llama", "llama"),
    "gemma": ("release_benchmark.methods.lvlm.gemma", "gemma"),
    "llava_next": ("release_benchmark.methods.lvlm.llava_next", "llava_next"),
}


def _resolve(entry):
    module_name, class_name = entry
    module = import_module(module_name)
    return getattr(module, class_name)


def get_train_method(name: str):
    if name == "resample":
        name = "erm"
    if name not in TRAIN_METHODS:
        raise KeyError(f"Unknown train method: {name}")
    return _resolve(TRAIN_METHODS[name])


def get_zeroshot_method(name: str):
    if name not in ZEROSHOT_METHODS:
        raise KeyError(f"Unknown zeroshot method: {name}")
    return _resolve(ZEROSHOT_METHODS[name])
