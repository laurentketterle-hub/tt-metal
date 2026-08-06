# Gemma-2 Common Utilities - Bounty tt-metal#51398
GEMMA2_CONFIGS = {
    "2b": {"hidden_size": 2048, "num_layers": 18, "num_heads": 8},
    "9b": {"hidden_size": 3584, "num_layers": 42, "num_heads": 16},
}

def create_gemma2_model(device, config_name="2b"):
    # Create Gemma-2 model on TT device.
    cfg = GEMMA2_CONFIGS[config_name]
    return None  # TODO: implement
