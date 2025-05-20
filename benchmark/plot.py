import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 20})
# plt.style.use("ggplot")

all_gpus = ["A10g", "L4", "L40S", "H200"]
systems = ["vllm", "TEI", "Arctic Inference"]
length_group_labels = ["50", "512"]

throughput_data_by_length = {
    "512": {  # Data for length 512
        "vllm": [195.71, 167.91, 380.69, 389.16],
        "TEI": [218.21, 190.99, 607.53, 1624.11],
        "Arctic Inference": [226.14, 173.96, 591.75, 1640.5],
    },
    "50": {  # Data for length 50
        "vllm": [59.02, 83.97, 82.84, 88.45],
        "TEI": [210.2, 196.43, 576.81, 618.82],
        "Arctic Inference": [194.32, 174.90, 558.67, 1419.65],
    },
}
# USD per second AWS on demand price
cost_per_gpu = {
    "H200": 10.60 / 3600,
    "L4": 1.67 / 3600,
    "A10g": 2.04 / 3600,
    "L40S": 3.77 / 3600,
    "T4": 0.98 / 3600,
}
# remove GPUs
GPUs_to_remove = set(["L4", "L40S"])
gpus = [gpu for gpu in all_gpus if gpu not in GPUs_to_remove]
for seq_len in length_group_labels:
    for sys in systems:
        # Create a new list excluding the GPUs to remove
        filtered_data = []
        for i_gpu, gpu in enumerate(all_gpus):  # Original GPU list
            if gpu not in GPUs_to_remove:
                print(f"keeping {i_gpu} {gpu} from {sys} {seq_len}")
                filtered_data.append(throughput_data_by_length[seq_len][sys][i_gpu])
        throughput_data_by_length[seq_len][sys] = filtered_data

n_gpus = len(gpus)
n_systems = len(systems)
n_length_groups = len(length_group_labels)

# Bar width and spacing
bar_width = 0.24  # Adjusted for clarity within groups
colors = [
    # "#FFFFFF", # Pure White
    "#F0F8FF",  # AliceBlue
    "#E6F2F8",  # Light Cyan Frost
    "#ADD8E6",  # LightBlue
    "#B0C4DE",  # LightSteelBlue
    "#D1D0CE",  # Silver Chalice
]
colors = [
    # "#F5FEFD", # Snowdrift White
    "#D6EAF8",  # Pale Blue Lily
    "#A9CCE3",  # Winter Blue
    "#7393B3",  # Steel Blue
    "#4A6A82",  # Shadowed Blue
    "#8395A7",  # Slate Gray
]
# colors = [
#     # "#FEFEFE", # Almost White
#     "#EBF4FA", # Glacier Blue
#     "#C1D9E9", # Serene Blue
#     # "#B4CBE0", # Ice Cap Blue
#     "#B0C4DE", # LightSteelBlue
#     "#C0C0C0", # Silver
# ]
# --- Calculate X-axis positions and labels ---
# Each primary group on the x-axis will be a (Length, GPU) combination
n_primary_x_groups = n_length_groups * n_gpus
x_indices = np.arange(n_primary_x_groups)  # Centers for each (Length, GPU) cluster
x_indices = []
for i in range(n_length_groups):
    for j in range(n_gpus):
        x_indices.append(i * (n_gpus + 1) + j)

xtick_labels = []
for length_label in length_group_labels:
    for gpu_name in gpus:
        xtick_labels.append(f"{gpu_name}")

cost_data_by_length = {}
for length_label in length_group_labels:
    cost_data_by_length[length_label] = {}
    for sys in systems:
        cost_data_by_length[length_label][sys] = []
        for i_gpu, gpu_name in enumerate(gpus):
            throughput = throughput_data_by_length[length_label][sys][i_gpu]
            cost_per_trillion_tokens = cost_per_gpu[gpu_name] / (throughput / 1000 / 1000 / 1000)
            cost_data_by_length[length_label][sys].append(cost_per_trillion_tokens)
            # print(
            #     f"{sys} {gpu_name} {length_label} {cost_per_billion_tokens} {throughput}"
            # )


def plot_data(data_dict, y_label, figname):
    fig, ax = plt.subplots(figsize=(16, 8))  # Wider figure for more groups

    legend_handles = {}  # To store unique handles for the legend (systems only)
    current_primary_x_group_idx = 0

    for length_label in length_group_labels:
        data_for_current_length = data_dict[length_label]
        for i_gpu, gpu_name in enumerate(gpus):
            # Center of the current (Length, GPU) primary group
            group_center_x = x_indices[current_primary_x_group_idx]

            for i_sys, sys_name in enumerate(systems):
                # Calculate offset for this sys's bar from the group_center_x
                # This centers the n_systems bars around group_center_x
                bar_offset = (i_sys - (n_systems - 1) / 2.0) * bar_width
                bar_x_position = group_center_x + bar_offset

                value = data_for_current_length[sys_name][i_gpu]

                # Add legend label only once per sys
                label_for_legend = None
                if sys_name not in legend_handles:
                    label_for_legend = sys_name

                bar_container = ax.bar(
                    bar_x_position,
                    value,
                    bar_width,
                    label=label_for_legend,
                    color=colors[i_sys],
                )
                if label_for_legend:  # Store one bar patch for the legend
                    legend_handles[sys_name] = bar_container[0]

            current_primary_x_group_idx += 1

    line_x = (x_indices[n_gpus - 1] + x_indices[n_gpus]) / 2
    shift = 0.8
    plt.axvline(x=line_x, color="black", linestyle="--")
    # Add light gray background to the left side of the plot
    ax.axvspan(xmin=-shift, xmax=line_x, color="#FEFBF6", alpha=0.5, zorder=-1)
    # Add a different color to the right side of the plot
    ax.axvspan(
        xmin=line_x, xmax=x_indices[-1] + shift, color="#F8F0E3", alpha=0.5, zorder=-1
    )

    # --- Setup Axes and Legend ---
    ax.set_ylabel(
        y_label,
        fontweight="bold",
    )
    if y_label == "Throughput (K tokens/s)":
        plt.text(
            x=-0.2,
            y=1600,
            s="Seq len 50",
            ha="center",
            va="center",
            fontweight="bold",
        )
        plt.text(
            x=line_x + 0.64,
            y=1600,
            s="Seq len 512",
            ha="center",
            va="center",
            fontweight="bold",
        )

        ax.annotate(
            "16x",
            xy=(0.72, 120),  # Lower point
            xytext=(0.72, 1480),  # Upper point
            textcoords="data",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                color="red",
                lw=2.4,
                connectionstyle="arc3,rad=0"  # Force straight line with rad=0
            ),
            horizontalalignment="center",  # Position text to the right of arrow
            verticalalignment="center",  # Center text vertically
            color="red",  # Text color matching the arrow
        )

        ax.annotate(
            "2.4x",
            xy=(1.02, 640),  # Lower point
            xytext=(1.02, 1480),  # Upper point
            textcoords="data",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                color="brown",
                lw=2.4,
                connectionstyle="arc3,rad=0"  # Force straight line with rad=0
            ),
            horizontalalignment="center",  # Position text to the right of arrow
            verticalalignment="center",  # Center text vertically
            color="brown",  # Text color matching the arrow
        )

        ax.annotate(
            "4.2x",
            xy=(3.72, 400),  # Lower point
            xytext=(3.72, 1698),  # Upper point
            textcoords="data",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                color="orange",
                lw=2.4,
                connectionstyle="arc3,rad=0"  # Force straight line with rad=0
            ),
            horizontalalignment="center",  # Position text to the right of arrow
            verticalalignment="center",  # Center text vertically
            color="orange",  # Text color matching the arrow
        )

        plt.ylim(0, 1800)
    elif y_label == "Cost ($/Trillion tokens)":
        plt.text(
            x=-0.2,
            y=32.8*1000,
            s="Seq len 50",
            ha="center",
            va="center",
            fontweight="bold",
        )
        plt.text(
            x=line_x + 0.64,
            y=32.8*1000,
            s="Seq len 512",
            ha="center",
            va="center",
            fontweight="bold",
        )

        ax.annotate(
            "16x",
            xy=(1.24, 1600),  # Upper point
            xytext=(1.24, 34800),  # Lower point
            textcoords="data",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                color="red",
                lw=2.4,
                connectionstyle="arc3,rad=0"  # Force straight line with rad=0
            ),
            horizontalalignment="center",  # Position text to the right of arrow
            verticalalignment="center",  # Center text vertically
            color="red",  # Text color matching the arrow
        )

        # ax.annotate(
        #     "2.4x",
        #     xy=(1.02, 5000),  # Lower point
        #     xytext=(1.02, 1480),  # Upper point
        #     textcoords="data",
        #     xycoords="data",
        #     arrowprops=dict(
        #         arrowstyle="<->",
        #         color="brown",
        #         lw=2.4,
        #         connectionstyle="arc3,rad=0"  # Force straight line with rad=0
        #     ),
        #     horizontalalignment="center",  # Position text to the right of arrow
        #     verticalalignment="center",  # Center text vertically
        #     color="brown",  # Text color matching the arrow
        # )

        ax.annotate(
            "4.2x",
            xy=(4.24, 1600),  # Lower point
            xytext=(4.24, 9000),  # Upper point
            textcoords="data",
            xycoords="data",
            arrowprops=dict(
                arrowstyle="<->",
                color="orange",
                lw=2.4,
                connectionstyle="arc3,rad=0"  # Force straight line with rad=0
            ),
            horizontalalignment="center",  # Position text to the right of arrow
            verticalalignment="center",  # Center text vertically
            color="orange",  # Text color matching the arrow
        )
        plt.ylim(0, 38000)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center")
    ax.tick_params(axis="y")
    ax.set_xlim(-shift, x_indices[-1] + shift)
    ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        # title="System",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.tight_layout(pad=2.5)  # Adjust padding

    plt.savefig(figname, dpi=300)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    plot_data(
        throughput_data_by_length,
        "Throughput (K tokens/s)",
        "embedding_throughput.png",
    )
    plot_data(cost_data_by_length, "Cost ($/Trillion tokens)", "embedding_cost.png")
