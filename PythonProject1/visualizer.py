import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TrainingVisualizer:
    def __init__(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.ax.set_title("Reward vs Episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.rewards = []

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(pady=5)

    def update(self, reward):
        self.rewards.append(reward)
        self.ax.clear()
        self.ax.plot(self.rewards, color="lime", linewidth=2)
        self.ax.set_title("Reward vs Episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas.draw()