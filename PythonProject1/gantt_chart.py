import matplotlib.pyplot as plt

def plot_gantt(schedule_log, num_machines):
    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for job, machine, start, end in schedule_log:
        ax.barh(
            machine,
            end - start,
            left=start,
            color=colors[job % len(colors)],
            edgecolor='black'
        )
        ax.text(
            start + (end - start) / 2,
            machine,
            f'J{job}',
            va='center',
            ha='center',
            color='white',
            fontsize=9
        )

    ax.set_yticks(range(num_machines))
    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")
    ax.set_title("DRL Job Shop Scheduling – Gantt Chart")

    plt.tight_layout()
    plt.show()


def plot_gantt_chart():
    return None