"""Strict checkpoint boundary for the frozen history-motion experiment only."""


def validate_residual_checkpoint(state: dict, expected: dict, args, recorded_args=None) -> None:
    prefix = "ball_velocity_head."
    source_head = {key for key in state if key.startswith(prefix)}
    target_head = {key for key in expected if key.startswith(prefix)}
    if not source_head and not target_head:
        return
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    evaluation = getattr(args, "require_stream25_checkpoint_contract", False)
    if unexpected or (source_head and missing) or (not source_head and missing != target_head):
        raise ValueError(f"Residual checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if evaluation and not source_head:
        raise ValueError("Residual evaluation requires a trained residual checkpoint, not baseline")
    if source_head:
        if recorded_args is None:
            raise ValueError("Residual checkpoint must record architecture args")
        for key, default in (("ball_velocity_history", True), ("ball_velocity_use_time", True),
                             ("ball_velocity_use_difference", True)):
            old = recorded_args.get(key, default) if isinstance(recorded_args, dict) else getattr(recorded_args, key, default)
            if old != getattr(args, key, default):
                raise ValueError(f"Residual checkpoint/config mismatch: {key}")
