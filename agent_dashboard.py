#!/usr/bin/env python3

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import json
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import os
from pathlib import Path

class AgentDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data_file = "logs/"  # Fixed: actual log directory
        self.agent_data = {}
        self.current_round = 0
        self.setup_layout()
        self.setup_callbacks()
        
        # Start background data monitoring
        self.monitor_thread = threading.Thread(target=self.monitor_agent_logs, daemon=True)
        self.monitor_thread.start()
    
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("🎯 DnD Auction Agent Dashboard", 
                   style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': 30}),
            
            # Control Panel
            html.Div([
                html.H3("⚙️ Agent Controls", style={'color': '#A23B72'}),
                html.Div([
                    html.Label("Extra Gold Multiplier for Bids:", 
                             style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Slider(
                        id='gold-multiplier-slider',
                        min=0.5,
                        max=5.0,
                        step=0.1,
                        value=1.0,
                        marks={i: f'{i}x' for i in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': 20}),
                
                html.Div([
                    html.Label("Minimum Gold Reserve:", 
                             style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Slider(
                        id='reserve-slider',
                        min=0,
                        max=50000,
                        step=1000,
                        value=5000,
                        marks={i: f'{i//1000}k' for i in range(0, 55000, 10000)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': 20}),
                
                html.Div([
                    html.Button('🔄 Apply Settings', id='apply-button', 
                              style={'backgroundColor': '#F18F01', 'color': 'white', 
                                   'border': 'none', 'padding': '10px 20px', 
                                   'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='settings-status', style={'marginLeft': 20, 'color': 'green'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'backgroundColor': '#F7F7F7', 'padding': 20, 'borderRadius': 10, 'marginBottom': 30}),
            
            # Real-time Metrics
            html.Div([
                html.H3("📊 Live Performance Metrics", style={'color': '#A23B72'}),
                html.Div(id='live-metrics', children=[
                    html.Div("Waiting for data...", style={'textAlign': 'center', 'fontSize': 18})
                ])
            ], style={'backgroundColor': '#F0F8FF', 'padding': 20, 'borderRadius': 10, 'marginBottom': 30}),
            
            # Plots
            html.Div([
                html.Div([
                    dcc.Graph(id='gold-progression-plot')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='points-progression-plot')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='bid-efficiency-plot')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='interest-utilization-plot')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),
            
            # Data storage
            dcc.Store(id='agent-data-store', data={}),
        ])
    
    def monitor_agent_logs(self):
        """Background thread to monitor agent log files"""
        while True:
            try:
                # Look for the latest agent log file
                log_dir = Path("logs")  # Fixed: correct log directory
                if log_dir.exists():
                    log_files = list(log_dir.glob("agent_local_rand_id_*.jsonl"))
                    if log_files:
                        latest_log = max(log_files, key=os.path.getctime)
                        self.parse_agent_log(latest_log)
                        print(f"📊 Monitoring log: {latest_log.name}")  # Debug info
                
                # Also check auction house logs
                auction_log = Path("dnd_auction_game/auction_house_log_2.jsonln")
                if auction_log.exists():
                    self.parse_auction_log(auction_log)
                    
            except Exception as e:
                print(f"Error monitoring logs: {e}")
            
            time.sleep(2)
    
    def parse_agent_log(self, log_file):
        """Parse agent log file for performance data"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            agent_name = "unknown"
            for line in lines[-50:]:  # Check recent lines
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        if 'name' in data:
                            agent_name = data['name']
                        # Extract relevant metrics
                        if agent_name not in self.agent_data:
                            self.agent_data[agent_name] = {
                                'rounds': [], 'gold': [], 'points': [], 'bids': []
                            }
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error parsing agent log: {e}")
    
    def parse_auction_log(self, log_file):
        """Parse auction house log for comprehensive data"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the latest rounds
            for line in lines[-10:]:  # Check last 10 rounds
                if line.strip():
                    try:
                        round_data = json.loads(line.strip())
                        self.current_round = round_data.get('round', 0)
                        
                        # Extract data for all agents
                        states = round_data.get('states', {})
                        for agent_id, state in states.items():
                            if agent_id not in self.agent_data:
                                self.agent_data[agent_id] = {
                                    'rounds': [], 'gold': [], 'points': [], 'bids': []
                                }
                            
                            # Update agent data
                            agent_data = self.agent_data[agent_id]
                            if self.current_round not in agent_data['rounds']:
                                agent_data['rounds'].append(self.current_round)
                                agent_data['gold'].append(state.get('gold', 0))
                                agent_data['points'].append(state.get('points', 0))
                        
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error parsing auction log: {e}")
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('agent-data-store', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_data_store(n):
            return self.agent_data
        
        @self.app.callback(
            Output('live-metrics', 'children'),
            Input('agent-data-store', 'data')
        )
        def update_live_metrics(data):
            if not data:
                return [html.Div("No data available", style={'textAlign': 'center'})]
            
            metrics = []
            for agent_id, agent_data in data.items():
                if agent_data['gold'] and agent_data['points']:
                    current_gold = agent_data['gold'][-1]
                    current_points = agent_data['points'][-1]
                    
                    # Determine agent type
                    agent_name = "Interest-Split Agent" if "359553" in agent_id else f"Agent {agent_id[-6:]}"
                    
                    metrics.append(
                        html.Div([
                            html.H4(agent_name, style={'margin': 0, 'color': '#2E86AB'}),
                            html.P(f"💰 Gold: {current_gold:,}", style={'margin': 5}),
                            html.P(f"🎯 Points: {current_points:,}", style={'margin': 5}),
                            html.P(f"📈 Efficiency: {current_points/(current_gold+1)*1000:.1f} pts/1k gold", 
                                  style={'margin': 5})
                        ], style={'backgroundColor': 'white', 'padding': 15, 'borderRadius': 8, 
                                'margin': 10, 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
                                'display': 'inline-block', 'minWidth': 200})
                    )
            
            return metrics
        
        @self.app.callback(
            Output('gold-progression-plot', 'figure'),
            Input('agent-data-store', 'data')
        )
        def update_gold_plot(data):
            fig = go.Figure()
            
            for agent_id, agent_data in data.items():
                if agent_data['rounds'] and agent_data['gold']:
                    agent_name = "Interest-Split Agent" if "359553" in agent_id else f"Agent {agent_id[-6:]}"
                    color = 'red' if "359553" in agent_id else 'blue'
                    
                    fig.add_trace(go.Scatter(
                        x=agent_data['rounds'],
                        y=agent_data['gold'],
                        mode='lines+markers',
                        name=agent_name,
                        line=dict(color=color, width=3 if "359553" in agent_id else 2)
                    ))
            
            fig.update_layout(
                title="💰 Gold Progression Over Time",
                xaxis_title="Round",
                yaxis_title="Gold Amount",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('points-progression-plot', 'figure'),
            Input('agent-data-store', 'data')
        )
        def update_points_plot(data):
            fig = go.Figure()
            
            for agent_id, agent_data in data.items():
                if agent_data['rounds'] and agent_data['points']:
                    agent_name = "Interest-Split Agent" if "359553" in agent_id else f"Agent {agent_id[-6:]}"
                    color = 'green' if "359553" in agent_id else 'orange'
                    
                    fig.add_trace(go.Scatter(
                        x=agent_data['rounds'],
                        y=agent_data['points'],
                        mode='lines+markers',
                        name=agent_name,
                        line=dict(color=color, width=3 if "359553" in agent_id else 2)
                    ))
            
            fig.update_layout(
                title="🎯 Points Progression Over Time",
                xaxis_title="Round",
                yaxis_title="Points (Victory Metric)",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('bid-efficiency-plot', 'figure'),
            Input('agent-data-store', 'data')
        )
        def update_efficiency_plot(data):
            fig = go.Figure()
            
            agents = []
            efficiency = []
            
            for agent_id, agent_data in data.items():
                if agent_data['gold'] and agent_data['points']:
                    current_gold = agent_data['gold'][-1]
                    current_points = agent_data['points'][-1]
                    eff = current_points / (current_gold + 1) * 1000
                    
                    agent_name = "Interest-Split" if "359553" in agent_id else f"Agent {agent_id[-6:]}"
                    agents.append(agent_name)
                    efficiency.append(eff)
            
            if agents:
                colors = ['red' if 'Interest-Split' in name else 'lightblue' for name in agents]
                fig.add_trace(go.Bar(x=agents, y=efficiency, marker_color=colors))
            
            fig.update_layout(
                title="📊 Bid Efficiency (Points per 1000 Gold)",
                xaxis_title="Agent",
                yaxis_title="Points per 1000 Gold",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('interest-utilization-plot', 'figure'),
            Input('agent-data-store', 'data')
        )
        def update_utilization_plot(data):
            # Calculate interest utilization metrics
            fig = go.Figure()
            
            # For now, show gold growth rate as a proxy for interest utilization
            for agent_id, agent_data in data.items():
                if len(agent_data['gold']) > 1:
                    gold_data = agent_data['gold']
                    rounds = agent_data['rounds']
                    
                    # Calculate growth rate
                    growth_rates = []
                    for i in range(1, len(gold_data)):
                        if gold_data[i-1] > 0:
                            growth = (gold_data[i] - gold_data[i-1]) / gold_data[i-1] * 100
                            growth_rates.append(growth)
                        else:
                            growth_rates.append(0)
                    
                    if growth_rates:
                        agent_name = "Interest-Split" if "359553" in agent_id else f"Agent {agent_id[-6:]}"
                        color = 'purple' if "359553" in agent_id else 'gray'
                        
                        fig.add_trace(go.Scatter(
                            x=rounds[1:],
                            y=growth_rates,
                            mode='lines',
                            name=f"{agent_name} Growth Rate",
                            line=dict(color=color, width=3 if "359553" in agent_id else 1)
                        ))
            
            fig.update_layout(
                title="📈 Gold Growth Rate (%)",
                xaxis_title="Round",
                yaxis_title="Growth Rate (%)",
                template="plotly_white",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('settings-status', 'children'),
            Input('apply-button', 'n_clicks'),
            Input('gold-multiplier-slider', 'value'),
            Input('reserve-slider', 'value')
        )
        def apply_settings(n_clicks, multiplier, reserve):
            if n_clicks:
                # Write settings to a config file that the agent can read
                settings = {
                    'gold_multiplier': multiplier,
                    'minimum_reserve': reserve,
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    with open('dnd_auction_agents/agent_config.json', 'w') as f:
                        json.dump(settings, f, indent=2)
                    
                    return f"✅ Settings applied: {multiplier}x multiplier, {reserve:,} reserve"
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            return ""

    def run(self, host='127.0.0.1', port=8050, debug=True):
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    dashboard = AgentDashboard()
    print("🚀 Starting Agent Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("⚙️ Use the controls to adjust agent parameters in real-time!")
    dashboard.run()