from flask import Flask, jsonify, send_from_directory, request as flask_request
from flask_cors import CORS
import time
import random
import math
import requests as req_lib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simulation State
class NetworkState:
    def __init__(self):
        self.phase = 0.0
        # Data history for moving averages or trends could be added here
        self.last_update = time.time()
        self.config = {
            "voip_alloc": 50,
            "threshold": 0.4,
            "ftp_prio": "std"
        }
        self.dataset = [] # Stores history for "Excel" view
        self.last_minute = None # Track sampling interval
        
    def get_current_metrics(self):
        now = time.time()
        dt = now - self.last_update
        self.phase += dt * 0.2  # Speed of cycle
        self.last_update = now
        
        # 1. Simulate Traffic Sources (incorporating a sine wave for peak/off-peak cycles)
        # Base load + sine wave + noise
        cycle = math.sin(self.phase) # -1 to 1
        
        # VoIP (Kbps): Fairly steady, slight increase during peaks
        voip_kbps = max(50, 120 + 40 * cycle + random.uniform(-10, 10))
        
        # HTTP (Mbps): Highly variable, follows daily cycle strongly
        http_mbps = max(1.0, 4.0 + 3.0 * cycle + random.uniform(-0.5, 0.5))
        
        # FTP (Mbps): Bursty, less dependent on cycle
        ftp_mbps = max(0.1, 2.0 + 1.0 * math.sin(self.phase * 0.3) + random.uniform(-0.5, 1.5))

        # Total load proxy (Mbps)
        total_load = (voip_kbps / 1000.0) + http_mbps + ftp_mbps
        
        # 2. Traffic Monitoring / Feature Extraction
        # Arrival rate scales with total load
        arrival_rate = int(total_load * 120 + random.uniform(-50, 50))
        
        # Base delay increases exponentially as load approaches capacity (Capacity ~ 12 Mbps)
        capacity = 12.0
        utilization_ratio = min(0.99, total_load / capacity)
        base_delay = 5.0 / (1.0 - utilization_ratio) + random.uniform(-2, 2)
        
        # Queue length builds up when load is high
        queue_length = int(max(0, (total_load - 5.0) * 50 + random.uniform(0, 20)))
        
        # 3. ML for Traffic Network (Simulated Logic)
        confidence = round(random.uniform(85.0, 98.0), 1)
        infer_time = random.randint(5, 15)
        
        if utilization_ratio < self.config['threshold']:
            state = "low"
        elif utilization_ratio < min(0.95, self.config['threshold'] + 0.35):
            state = "med"
        else:
            state = "high"

        # 4. Adaptive QoS Controller (Resource Allocation)
        if state == "low":
            # Normal allocation
            bw_voip, bw_http, bw_ftp = 20, 50, 30
            # Priorities 1, 2, 3 normal
            q_voip, q_http, q_ftp = 10, 30, 60
        elif state == "med":
            # Protect VoIP and HTTP slightly
            bw_voip, bw_http, bw_ftp = 30, 45, 25
            q_voip, q_http, q_ftp = 20, 40, 40
            # Apply FTP Priority Logic
            if self.config['ftp_prio'] == "high": bw_ftp += 10; bw_http -= 10; q_ftp -= 10; q_http += 10
            if self.config['ftp_prio'] == "low": bw_ftp -= 10; bw_http += 10; q_ftp += 10; q_http -= 10
        else:
            # High congestion: Heavily prioritize VoIP based on config
            bw_voip = self.config['voip_alloc']
            rem = 100 - bw_voip
            bw_http = int(rem * 0.7)
            bw_ftp = rem - bw_http
            
            # Queue occupancy shifted to protect VoIP
            q_voip, q_http, q_ftp = 50, 30, 20
            if self.config['ftp_prio'] == "high": q_ftp -= 10; q_http += 10

        # 5. Bottleneck Router / Performance Analyzer
        link_utilization = min(100, int(utilization_ratio * 100))
        queue_occupancy = min(100, int((queue_length / 500.0) * 100))
        
        # Post-QoS Performance Metrics
        if state == "high":
            # QoS is acting to save VoIP but overall delay is high, loss happens mainly on FTP
            final_delay = base_delay * 0.8  # QoS managed it slightly better than raw
            packet_loss = 0.5 + (utilization_ratio - 0.75) * 5.0 + random.uniform(0, 0.5)
        elif state == "med":
            final_delay = base_delay * 0.9
            packet_loss = random.uniform(0.05, 0.3)
        else:
            final_delay = base_delay
            packet_loss = random.uniform(0.01, 0.05)
            
        final_tput = total_load * (1.0 - packet_loss/100.0)

        # Generate Alerts based on state transitions or thresholds
        alerts = []
        now_str = time.strftime("%H:%M:%S")
        if utilization_ratio > 0.85:
            alerts.append({"time": now_str, "msg": "Critical link utilization detected > 85%", "cls": "crit"})
        elif state == "high":
            alerts.append({"time": now_str, "msg": f"QoS active: VoIP bandwidth increased to {bw_voip}%", "cls": "warn"})
        elif state == "low" and random.random() < 0.1:
            alerts.append({"time": now_str, "msg": "Nominal traffic conditions - Policy reset", "cls": "ok"})
            
        if packet_loss > 1.0:
             alerts.append({"time": now_str, "msg": f"Packet loss elevated ({packet_loss:.1f}%) on background queues", "cls": "warn"})

        metrics_data = {
            "traffic": {
                "voip": round(voip_kbps),
                "http": round(http_mbps, 1),
                "ftp": round(ftp_mbps, 1),
                "aggregate": round(total_load, 1) # for the sparkline
            },
            "monitoring": {
                "arrival_rate": arrival_rate,
                "delay": round(base_delay, 1),
                "queue_length": queue_length
            },
            "ml": {
                "state": state,
                "confidence": confidence,
                "infer_time": infer_time
            },
            "qos": {
                "bandwidth": {"voip": bw_voip, "http": bw_http, "ftp": bw_ftp},
                "queues": {"voip": q_voip, "http": q_http, "ftp": q_ftp}
            },
            "router": {
                "link_utilization": link_utilization,
                "queue_occupancy": queue_occupancy
            },
            "performance": {
                "delay": round(final_delay, 1),
                "throughput": round(final_tput, 2),
                "packet_loss": round(packet_loss, 2)
            },
            "alerts": alerts
        }
        
        # Append to dataset history at 1-minute intervals
        current_min = time.strftime("%H:%M")
        if current_min != self.last_minute:
            snapshot = {
                "time": now_str,
                "voip": round(voip_kbps),
                "http": round(http_mbps, 1),
                "ftp": round(ftp_mbps, 1),
                "delay": round(final_delay, 1),
                "throughput": round(final_tput, 2),
                "loss": round(packet_loss, 2),
                "state": state
            }
            self.dataset.insert(0, snapshot) # Latest first
            if len(self.dataset) > 100:
                self.dataset.pop()
            self.last_minute = current_min
            
        return metrics_data

state_manager = NetworkState()

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(state_manager.get_current_metrics())

@app.route('/api/towers', methods=['GET'])
def get_towers():
    """Proxy OpenCelliD so browser avoids CORS."""
    lat  = flask_request.args.get('lat', type=float)
    lng  = flask_request.args.get('lng', type=float)
    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng required'}), 400

    OCID_KEY = 'pk.67c74360612eba39b08f928817786da9'
    delta    = 0.008         # ~800m radius (safely under 4,000,000 sq. mts limit)
    bbox     = f"{lat-delta},{lng-delta},{lat+delta},{lng+delta}"
    url      = f"https://opencellid.org/cell/getInArea?key={OCID_KEY}&BBOX={bbox}&format=json&limit=500"

    try:
        resp = req_lib.get(url, timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'error': str(e), 'cells': []}), 502


@app.route('/')
def index():
    import os
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/api/config', methods=['POST'])
def update_config():
    from flask import request
    data = request.json
    if not data:
        return jsonify({"status": "error", "msg": "No data provided"}), 400
    
    # Update state manager config
    if 'voip_alloc' in data: state_manager.config['voip_alloc'] = int(data['voip_alloc'])
    if 'threshold' in data: state_manager.config['threshold'] = float(data['threshold'])
    if 'ftp_prio' in data: state_manager.config['ftp_prio'] = data['ftp_prio']
    
    return jsonify({"status": "success", "config": state_manager.config})

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    return jsonify(state_manager.dataset)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
