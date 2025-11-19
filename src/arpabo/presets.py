"""Preset configurations for common use cases."""

PRESETS = {
    "first-pass": {
        "description": "Optimized for first-pass diversity before LLM rescoring",
        "smoothing_method": "good_turing",
        "recommended_order": 2,
        "rationale": "Good-Turing smoothing with bigrams encourages diverse N-best lists for LLM to rerank",
        "use_case": "First-pass ASR decoding before neural LM rescoring",
    },
    "rescoring": {
        "description": "Optimized for accurate rescoring (sharp predictions)",
        "smoothing_method": "kneser_ney",
        "recommended_order": 4,
        "rationale": "Kneser-Ney with higher-order context provides best discrimination between hypotheses",
        "use_case": "N-best list rescoring or reranking",
    },
    "balanced": {
        "description": "Balanced configuration for general use",
        "smoothing_method": "kneser_ney",
        "recommended_order": 3,
        "rationale": "Good balance between speed, memory, and accuracy",
        "use_case": "General-purpose speech recognition or text prediction",
    },
    "fast": {
        "description": "Fast decoding with minimal memory",
        "smoothing_method": "good_turing",
        "recommended_order": 2,
        "rationale": "Bigrams are fast to query and use minimal memory",
        "use_case": "Mobile devices or real-time applications",
    },
    "accurate": {
        "description": "Maximum accuracy regardless of speed",
        "smoothing_method": "kneser_ney",
        "recommended_order": 5,
        "rationale": "Higher order provides maximum context for best predictions",
        "use_case": "Offline batch processing where accuracy is critical",
    },
}


def get_preset(preset_name: str) -> dict:
    """
    Get preset configuration by name.

    Args:
        preset_name: Name of preset

    Returns:
        Dictionary with preset configuration

    Raises:
        ValueError: If preset_name is unknown

    Example:
        config = get_preset("first-pass")
        print(config["smoothing_method"])  # "good_turing"
    """
    if preset_name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: {available}")

    return PRESETS[preset_name].copy()


def list_presets() -> list[str]:
    """
    List available preset names.

    Returns:
        Sorted list of preset names

    Example:
        presets = list_presets()
        for name in presets:
            config = get_preset(name)
            print(f"{name}: {config['description']}")
    """
    return sorted(PRESETS.keys())


def print_presets() -> None:
    """
    Print formatted table of all available presets.

    Example:
        print_presets()
    """
    print("\nAvailable Presets")
    print("=" * 80)
    print()

    for name in sorted(PRESETS.keys()):
        config = PRESETS[name]
        print(f"Preset: {name}")
        print("-" * 80)
        print(f"  Description:  {config['description']}")
        print(f"  Smoothing:    {config['smoothing_method']}")
        print(f"  Order:        {config['recommended_order']}")
        print(f"  Use case:     {config['use_case']}")
        print(f"  Rationale:    {config['rationale']}")
        print()
