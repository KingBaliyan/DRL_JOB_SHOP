import tkinter as tk
from env_jobshop import JobShopEnv

env = JobShopEnv()
state, _ = env.reset()

root = tk.Tk()
root.title("Job Shop Step Viewer")

log = tk.Text(root, width=60, height=20)
log.pack()

def step():
    global state
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    log.insert(tk.END, f"Job {action} → Reward {reward}\n")
    if done:
        log.insert(tk.END, "All jobs finished\n")

tk.Button(root, text="Next Step", command=step).pack()
root.mainloop()