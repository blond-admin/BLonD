def as_n_buckets(time_distance: float, f_rf: float) -> int:
    """Number of RF buckets corresponding to a physical time distance [s]."""
    return round(time_distance * f_rf)
