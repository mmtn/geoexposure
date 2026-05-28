# BARE MINIMUM:
# - window() without dwell times returns correct rows
# - window() with dwell times clips dwell correctly
# - remove_data_from_trajectory edge cases:
#     keep_end_chunks=True with two chunks available
#     target_fraction unreachable without dropping below 3 points
#     partial chunk at end
#     trajectory shorter than one chunk
