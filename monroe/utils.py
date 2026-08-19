from datetime import datetime

__all__ = ["printf", "count_parameters", "str2bool"]


def printf(message: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def count_parameters(model) -> tuple[int, int, int]:
    enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

    decoders = model.decoders
    if decoders is not None:
        total_dec_params = sum(p.numel() for p in decoders.parameters() if p.requires_grad)
        n_decs = getattr(decoders, "num_tasks", 1)
        avg_dec_params = total_dec_params // n_decs if n_decs > 0 else 0
    else:
        total_dec_params = 0
        n_decs = 0
        avg_dec_params = 0

    return enc_params, avg_dec_params, n_decs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")
