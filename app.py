from flask import Flask, render_template, jsonify, request, Response
from jobshop_drl import JobShopEnv, DQNAgent, train_agent, test_agent
import threading

app = Flask(__name__)

# Global variables to store training state
training_progress = []
is_training = False
schedule_result = []
baseline_schedule = []
baseline_makespan = 0
last_metrics_csv = ""
ai_logs = []

# Instantiate global environment and agent
env = JobShopEnv(data_file="factory_data.csv")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

def background_training(episodes):
    global training_progress, is_training, schedule_result, env, agent, ai_logs
    training_progress.clear()
    
    # Callback to push progress
    def progress_callback(episode, reward):
        training_progress.append({"episode": episode, "reward": reward})
        
    train_agent(env, agent, episodes=episodes, progress_callback=progress_callback)
    
    schedule_result, smart_logs = test_agent(env, agent)
    ai_logs.extend(["\n--- EXECUTING OPTIMAL SMART AI POLICY ---"] + smart_logs)
    is_training = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_training", methods=["POST"])
def start_training():
    global is_training, agent, env, schedule_result, baseline_schedule, baseline_makespan, ai_logs
    if is_training:
        return jsonify({"status": "already_training"})
        
    data = request.json or {}
    episodes = int(data.get("episodes", 400))
    lr = float(data.get("lr", 0.001))
    epsilon_decay = float(data.get("epsilon_decay", 0.995))
    simulate_breakdown = bool(data.get("breakdown", False))
    
    breakdown_prob = 0.15 if simulate_breakdown else 0.0
        
    # Reset agent and env for a fresh training session
    env = JobShopEnv(data_file="factory_data.csv", breakdown_prob=breakdown_prob)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, lr=lr, epsilon_decay=epsilon_decay)
    schedule_result = []
    
    # Compute Baseline (Random Dumb Schedule)
    dummy_env = JobShopEnv(data_file="factory_data.csv", breakdown_prob=breakdown_prob)
    baseline_schedule, dumb_logs = test_agent(dummy_env, agent, random_baseline=True)
    baseline_makespan = max([s["end"] for s in baseline_schedule]) if baseline_schedule else 0
    ai_logs = ["--- STARTING DUMB AI BASELINE ---"] + dumb_logs + ["\n--- TRAINING SMART AI ---"]
    
    is_training = True
    
    thread = threading.Thread(target=background_training, args=(episodes,))
    thread.start()
    return jsonify({"status": "started"})

@app.route("/get_progress")
def get_progress():
    return jsonify({
        "is_training": is_training,
        "progress": training_progress,
        "schedule": schedule_result,
        "baseline_makespan": baseline_makespan,
        "logs": ai_logs
    })

@app.route("/save_model", methods=["POST"])
def save_model():
    if agent:
        agent.save("dqn_model.pth")
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route("/load_model", methods=["POST"])
def load_model():
    if agent and agent.load("dqn_model.pth"):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route("/get_factory_data")
def get_factory_data():
    try:
        with open("factory_data.csv", "r") as f:
            return jsonify({"csv": f.read()})
    except:
        return jsonify({"csv": ""})

@app.route("/save_factory_data", methods=["POST"])
def save_factory_data():
    csv_data = request.json.get("csv", "")
    with open("factory_data.csv", "w") as f:
        f.write(csv_data)
    return jsonify({"status": "success"})

@app.route("/download_report")
def download_report():
    csv_data = "Job,Machine,Start_Time,End_Time\n"
    for s in schedule_result:
        csv_data += f"{s['job']},{s['machine']},{s['start']},{s['end']}\n"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=AI_Schedule_Report.csv"}
    )

if __name__ == "__main__":
    print("Dashboard running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
