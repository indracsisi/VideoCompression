import math

def compute_final_score_ratio(C, V, vmaf_threshold, compression_weight=0.7,
                              quality_weight=0.3, soft_threshold_margin=5.0):
    """
    Compute the final compression score given:
        C  – compression ratio (compressed_size / original_size), 0 < C <= 1
        V  – VMAF quality score (0–100)
        vmaf_threshold – minimum acceptable VMAF
        compression_weight, quality_weight – should sum to 1.0
        soft_threshold_margin – margin below the threshold for the “soft zone”
    Returns a score in [0.0, 1.0].
    """
    # Hard cutoff: quality too low
    hard_cutoff = vmaf_threshold - soft_threshold_margin
    if V < hard_cutoff:
        return 0.0

    # Soft zone: V is close to threshold
    if V < vmaf_threshold:
        # Quadratic quality factor rising from 0 to 0.7
        soft_pos = (V - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_pos ** 2)

        # Compression component based on the size ratio C
        if C >= 0.95:  # less than ~1.05× compression
            compression_component = 0.0
        else:
            ratio = 1.0 / C  # convert to “times compressed”
            if ratio <= 20:
                compression_component = ((ratio - 1) / 19) ** 1.5
            else:
                compression_component = 1.0 + 0.3 * math.log(ratio / 20.0)
            compression_component = min(1.3, compression_component)

        return min(1.0, compression_component * quality_factor)

    # Above threshold: V meets or exceeds the required quality
    vmaf_excess = V - vmaf_threshold
    max_excess = 100.0 - vmaf_threshold
    # Linear quality component from 0.7 at threshold to 1.0 at V=100
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_excess)

    # Compression component in the high-quality zone
    if C >= 0.95:
        compression_component = 0.0
    elif C >= 0.80:
        # Poor compression: small quadratic reward
        ratio = 1.0 / C
        compression_component = (ratio - 1.0) ** 2 * 0.4
    else:
        ratio = 1.0 / C
        if ratio <= 20:
            compression_component = ((ratio - 1.25) / 18.75) ** 1.2 + 0.025
        else:
            compression_component = 1.0 + 0.3 * math.log(ratio / 20.0)
        compression_component = min(1.3, compression_component)

    # Weighted sum of compression and quality contributions, capped at 1.0
    final_score = (compression_weight * compression_component +
                   quality_weight * quality_component)
    return min(1.0, final_score)
