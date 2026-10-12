from TMD4.native_giga_model import create_native_giga_policy


def create_tmd(**kwargs):
    return create_native_giga_policy(**kwargs)


__all__ = ["create_tmd"]
