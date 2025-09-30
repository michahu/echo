# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List
import matplotlib.ticker as mticker
import pandas as pd

# %%
efficiency = pd.read_csv("peoplejoin/workspace/stateful-no-history/driving_school_task_efficiency.csv")
overlap = pd.read_csv("peoplejoin/workspace/stateful-no-history/driving_school_overlap.csv")

# %%
def get_data(input_pth):

    efficiency_data = np.genfromtxt(os.path.join(input_pth, "task_efficiency.csv"), delimiter=',')

    print("Efficiency")
    print(f"Mean: {np.mean(efficiency_data, axis=0)}")
    print(f"Std: {np.std(efficiency_data, axis=0)}")

    overlap_data = np.genfromtxt(os.path.join(input_pth, "overlaps.csv"), delimiter=',')

    print("Overlaps")
    print(f"Mean: {np.mean(overlap_data, axis=0)}")
    print(f"Std: {np.std(overlap_data, axis=0)}")

    return efficiency_data, overlap_data

# %%
from scipy.stats import rankdata

def rank_arrays_by_element(arrays: List[np.ndarray], method: str) -> List[np.ndarray]:
    """
    Ranks elements across multiple 2D arrays at each corresponding position.

    This function takes a list of 'k' 2D NumPy arrays, which must all have
    the same dimensions. For each element position (row, col), it looks at the
    'k' values across all arrays at that position and assigns a rank from 1 to 'k'.

    In the case of a tie, this function consistently assigns the same minimum
    rank to all tied values. The rank for the next element is then incremented
    accordingly. For example, if two values are tied for the smallest, they both
    receive a rank of 1, and the next highest value receives a rank of 3.

    Args:
        arrays: A list of 2D NumPy arrays. All arrays in the list must
                have the exact same shape.

    Returns:
        A new list of 'k' 2D NumPy arrays of the same shape as the input
        arrays, where each element is the rank of the corresponding input
        element.

    Raises:
        ValueError: If the input list of arrays is empty or if the arrays
                    do not all have the same shape.
        ImportError: If SciPy is not installed, as it is required for ranking.
    """
    # --- 1. Input Validation ---
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")

    first_shape = arrays[0].shape
    if not all(arr.shape == first_shape for arr in arrays):
        # trim to dimension of the smallest array
        min_shape = min(arr.shape for arr in arrays)
        arrays = [arr[:min_shape[0], :min_shape[1]] for arr in arrays]

    # --- 2. Stack and Prepare Data ---
    # Stack the k arrays into a single 3D array of shape (k, rows, cols).
    stacked_arrays = np.stack(arrays, axis=0)

    # --- 3. Perform Ranking with 'min' Tie-breaking ---
    # We use `scipy.stats.rankdata` with `method='min'` to handle ties.
    # This is applied to each 1D slice along the first axis (the 'k' dimension)
    # for every (row, col) position.

    # `np.apply_along_axis` iterates over each pixel stack and applies the ranking.
    ranks = np.apply_along_axis(
        func1d=lambda x: rankdata(x, method=method),
        axis=0,
        arr=stacked_arrays
    ).astype(int)  # rankdata returns floats, so we cast the result to int.

    # --- 4. Format and Return Output ---
    # Convert the 3D ranks array back into a list of 2D arrays.
    return [arr for arr in ranks]

# %%
input_pths = [
    "peoplejoin/workspace/stateful-no-history/awm/",
    "peoplejoin/workspace/stateful-no-history/gce/",
    "peoplejoin/workspace/stateful-no-history/reflexion/",
    "peoplejoin/workspace/stateful-no-history/baseline/",
]
efficiency_data = []
overlap_data = []
for input_pth in input_pths:
    eff_data, ov_data = get_data(input_pth)
    efficiency_data.append(eff_data[:, 1:])
    overlap_data.append(ov_data[:, 1:])

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(np.mean(overlap_data[0], axis=0), label='AWM', marker='o')
ax.plot(np.mean(overlap_data[1], axis=0), label='GCE', marker='s')
ax.plot(np.mean(overlap_data[2], axis=0), label='Reflexion', marker='^')
ax.plot(np.mean(overlap_data[3], axis=0), label='Baseline', marker='d', linestyle='--')

# --- 3. Customizing the Axes (User Requests) ---
# Set labels and title for clarity
ax.set_xlabel('Task Index')
ax.set_ylabel('Rank')
ax.set_title('Task Performance')

# Request 2: Make the y-axis ticks whole numbers.
# We use MaxNLocator with the 'integer' parameter to ensure only integer ticks.
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Add a legend to identify the lines
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(np.mean(efficiency_data[0], axis=0), label='AWM', marker='o')
ax.plot(np.mean(efficiency_data[1], axis=0), label='GCE', marker='s')
ax.plot(np.mean(efficiency_data[2], axis=0), label='Reflexion', marker='^')
ax.plot(np.mean(efficiency_data[3], axis=0), label='Baseline', marker='d', linestyle='--')

# --- 3. Customizing the Axes (User Requests) ---
# Set labels and title for clarity
ax.set_xlabel('Task Index')
ax.set_ylabel('Rank')
ax.set_title('Task Efficiency')

ax.invert_yaxis()

# Request 2: Make the y-axis ticks whole numbers.
# We use MaxNLocator with the 'integer' parameter to ensure only integer ticks.
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Add a legend to identify the lines
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability

# %%
ranks = rank_arrays_by_element(efficiency_data, 'min')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(np.mean(ranks[0], axis=0), label='AWM', marker='o')
ax.plot(np.mean(ranks[1], axis=0), label='GCE', marker='s')
ax.plot(np.mean(ranks[2], axis=0), label='Reflexion', marker='^')
ax.plot(np.mean(ranks[3], axis=0), label='Baseline', marker='d', linestyle='--')

# --- 3. Customizing the Axes (User Requests) ---
# Set labels and title for clarity
ax.set_xlabel('Task Index')
ax.set_ylabel('Rank')
ax.set_title('Task Efficiency Ranks')

# Request 1: Invert the y-axis so that rank 1 is at the top.
ax.invert_yaxis()

# Request 2: Make the y-axis ticks whole numbers.
# We use MaxNLocator with the 'integer' parameter to ensure only integer ticks.
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Add a legend to identify the lines
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability

# %%
perf_ranks = rank_arrays_by_element(overlap_data, 'max')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(np.mean(perf_ranks[0], axis=0), label='AWM', marker='o')
ax.plot(np.mean(perf_ranks[1], axis=0), label='GCE', marker='s')
ax.plot(np.mean(perf_ranks[2], axis=0), label='Reflexion', marker='^')
ax.plot(np.mean(perf_ranks[3], axis=0), label='Baseline', marker='d', linestyle='--')

# --- 3. Customizing the Axes (User Requests) ---
# Set labels and title for clarity
ax.set_xlabel('Task Index')
ax.set_ylabel('Rank')
ax.set_title('Performance Ranks')

# Request 2: Make the y-axis ticks whole numbers.
# We use MaxNLocator with the 'integer' parameter to ensure only integer ticks.
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Add a legend to identify the lines
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6) # Add grid for better readability

# %%
perf_ranks = rank_arrays_by_element(overlap_data, 'max')

# %%
perf_ranks[0]

# %%
perf_ranks[1]

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data extracted from the image, disregarding standard deviations.
data = {
    'Algorithm': [
        'Baseline, function calling', 'Reflexion', 'AWM', 'GCE', 'GCE, conterfactuals only',
        'Baseline, function calling', 'Reflexion', 'AWM', 'GCE', 'GCE, conterfactuals only',
        'Baseline, function calling', 'Reflexion', 'AWM', 'GCE', 'GCE, conterfactuals only',
        'Baseline, function calling', 'Reflexion', 'AWM', 'GCE', 'GCE, conterfactuals only',
        'Baseline, function calling', 'Reflexion', 'AWM', 'GCE', 'GCE, conterfactuals only'
    ],
    'Timestep': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    'Score': [
        48.52941176, 55, 51.92307692, 60.9375, 57.37,
        57.35294118, 36.67, 38.46153846, 53.125, 58.19,
        35.29411765, 25, 36.53846154, 53.125, 58.19,
        30.88235294, 25, 25, 46.875, 58.19,
        27.94117647, 20, 34.61538462, 54.6875, 59.01
    ],
    'Messages': [
        11.58, 12.6, 11.11538462, 11.125, 10.73,
        12.05, 8.6, 8.653846154, 11.953125, 11.63,
        7.73, 8.16, 7.84615385, 10.984375, 11.63,
        6.64, 5.7, 4.80769231, 9.203125, 10.42,
        6.26, 6.03, 5.46153846, 8.9375, 9.7
    ]
}

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Set the style for the plots
sns.set_theme(style="whitegrid")

# Create a figure and a set of subplots
# We will have 2 rows, 1 column of plots
fig, axes = plt.subplots(2, 1, figsize=(16, 16))
fig.suptitle('Algorithm Performance Over Timesteps', fontsize=20)

# --- Plot 1: Scores ---
sns.lineplot(
    ax=axes[0],
    data=df,
    x='Timestep',
    y='Score',
    hue='Algorithm',
    marker='o', # Add markers to the points
    linewidth=2.5
)
axes[0].set_title('Scores per Timestep', fontsize=16)
axes[0].set_xlabel('Timestep', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(True) # Ensure grid is visible
# Set x-axis to only show integer timesteps
axes[0].set_xticks(df['Timestep'].unique())


# --- Plot 2: Messages ---
sns.lineplot(
    ax=axes[1],
    data=df,
    x='Timestep',
    y='Messages',
    hue='Algorithm',
    marker='o', # Add markers to the points
    linewidth=2.5
)
axes[1].set_title('Messages per Timestep', fontsize=16)
axes[1].set_xlabel('Timestep', fontsize=12)
axes[1].set_ylabel('Messages', fontsize=12)
axes[1].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True) # Ensure grid is visible
# Set x-axis to only show integer timesteps
axes[1].set_xticks(df['Timestep'].unique())


# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust rect to make space for suptitle and legends

# Display the plots
plt.show()


# %%



