# job_flow_view.py
import tkinter as tk

class JobFlowView:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, width=500, height=250, bg="#020617")
        self.canvas.pack(pady=10)

        self.machines = []
        for i in range(3):
            x = 80 + i * 150
            rect = self.canvas.create_rectangle(x, 50, x+100, 120, fill="#1e293b")
            text = self.canvas.create_text(x+50, 85, text=f"Machine {i}", fill="white")
            self.machines.append(rect)

        self.job = self.canvas.create_oval(30, 160, 60, 190, fill="orange")

    def move_job(self, machine_id):
        x = 80 + machine_id * 150
        self.canvas.coords(self.job, x+30, 160, x+60, 190)