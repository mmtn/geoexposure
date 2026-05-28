"""Constants to define strings for reused column names."""

# Input columns
DATETIME = "datetime"
X = "x"
Y = "y"

# Computed columns
DWELL_TIME_SECONDS = "dwell_time_seconds"
DWELL_FORWARD = "_dwell_forward"
DWELL_BACKWARD = "_dwell_backward"

# Convenience variables
REQUIRED_COLUMNS = (DATETIME, X, Y)
REQUIRED_COLS_LIST = list(REQUIRED_COLUMNS)
DWELL_COLUMNS = (DWELL_TIME_SECONDS, DWELL_FORWARD, DWELL_BACKWARD)
