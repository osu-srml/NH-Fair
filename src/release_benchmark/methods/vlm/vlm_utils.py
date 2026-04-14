"""Prompt utilities for CLIP / BLIP2 VLM methods."""


def set_matching_prompt(args):
    """Return CLIP-style text prompts for zero-shot matching."""
    prompts = {
        "celeba": [
            "A photo of a person with non-wavy hair.",
            "A photo of a person with wavy hair.",
        ],
        "facet": [
            "A photo of a person with non-visible face.",
            "A photo of a person with visible face.",
        ],
        "waterbirds": ["A photo of a landbird.", "A photo of a waterbird."],
        "fairface": [
            "A photo of a White person.",
            "A photo of a Black person.",
            "A photo of a Latino or Hispanic person.",
            "A photo of an East Asian person.",
            "A photo of a Southeast Asian person.",
            "A photo of an Indian person.",
            "A photo of a Middle Eastern person.",
        ],
    }
    if args.dataset == "utk":
        if args.sa == "race":
            return ["A photo of a male.", "A photo of a female."]
        return ["A photo of a Non-White person.", "A photo of a White person."]
    texts = prompts.get(args.dataset)
    if texts is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return texts
