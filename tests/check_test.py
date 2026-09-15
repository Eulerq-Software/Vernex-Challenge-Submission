from waveform_optimization.core import islr_from_phase_indices

# Binary Barker sequence: + + + - + - -
q = [0, 0, 0, 1, 0, 1, 1]

print(islr_from_phase_indices(q, Q=2))


# import os

# print("Workspace folder:", os.environ.get("WORKSPACE_FOLDER"))