"""
Author: Douwe Berkeij
Date: 22-10-2025
"""

import os
import json
import matplotlib.pyplot as plt

def extract_points_from_line(record_line, lookup_agent_id):
    try:
        data = json.loads(record_line)
    except json.JSONDecodeError:
        return None

    # states is a dict mapping agent_id -> {"gold": .., "points": ..}
    states = data.get("states") or {}
    if lookup_agent_id and lookup_agent_id in states:
        return states[lookup_agent_id].get("points")
    # fall back to current_agent if lookup_agent_id not provided/found
    current = data.get("current_agent")
    if current and current in states:
        return states[current].get("points")
    return None

def plot_points_over_rounds(points_by_round, agent_id):
    rounds = list(range(1, len(points_by_round) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, points_by_round, marker='o')
    plt.title(f'Points Over Rounds for Agent {agent_id}')
    plt.xlabel('Round')
    plt.ylabel('Points')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_dir = "dnd_auction_agents/logs"

    temp_list = []

    for log_file in os.listdir(log_dir):
        if not log_file.endswith(".jsonl"):
            continue

        # filename format expected: agent_<agentid>_nX.jsonl
        parts = log_file.split("_")
        agent_id = parts[1] if len(parts) > 1 else None

        log_path = os.path.join(log_dir, log_file)
        points_by_round = []

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pts = extract_points_from_line(line, agent_id)
                points_by_round.append(pts)

        if len(points_by_round) >= 999:
            plot_points_over_rounds(points_by_round, "Dutch Courage")
        else:
            temp_list += points_by_round
            if len(temp_list) >= 900:
                plot_points_over_rounds(temp_list, "Dutch Courage")