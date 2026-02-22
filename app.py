from flask import Flask, jsonify, send_from_directory, request as flask_request
from flask_cors import CORS
import time
import random
import math
import requests as req_lib
import os
import psycopg2
import psutil
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()  # Load environment variables before initializing classes

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
        self.last_minute = None # Track sampling interval
        
        # Real Traffic Baseline
        self.last_net_io = psutil.net_io_counters()
        self.capacity_mbps = 100.0  # Max assumed local capacity for utilization calcs
        
        # Initialize Supabase DB connection
        self.db_url = os.environ.get("SUPABASE_URL")
        self._init_db()

    def _get_db_connection(self):
        if not self.db_url:
            return None
        try:
            return psycopg2.connect(self.db_url)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def _init_db(self):
        conn = self._get_db_connection()
        if not conn: return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS network_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        time_str VARCHAR(10),
                        voip_kbps INTEGER,
                        http_mbps FLOAT,
                        ftp_mbps FLOAT,
                        delay_ms FLOAT,
                        throughput_gbps FLOAT,
                        packet_loss_pct FLOAT,
                        state VARCHAR(10)
                    );
                """)
            conn.commit()
        except Exception as e:
            print(f"Error initializing database table: {e}")
        finally:
            conn.close()
        
    def get_current_metrics(self):
        now = time.time()
        dt = now - self.last_update
        if dt == 0: dt = 0.001
        self.last_update = now
        
        # 1. Real Network Traffic Stats via psutil
        current_net_io = psutil.net_io_counters()
        
        bytes_sent = current_net_io.bytes_sent - self.last_net_io.bytes_sent
        bytes_recv = current_net_io.bytes_recv - self.last_net_io.bytes_recv
        self.last_net_io = current_net_io
        
        # Total load in Mbps
        total_bytes = bytes_sent + bytes_recv
        total_load_mbps = (total_bytes * 8) / (dt * 1000000)
        
        # Floor tiny background noise to keep dashboard clean and avoid 0 errors
        total_load = max(0.01, total_load_mbps)

        # Distribute real load proportionally into our standard traffic bins for UI
        # Baseline noise usually hits VoIP, heavy hits HTTP/Video
        if total_load < 1.0:
            voip_kbps = max(20, total_load * 1000 * 0.4)
            http_mbps = max(0.1, total_load * 0.5)
            ftp_mbps = max(0.0, total_load * 0.1)
        else:
            voip_kbps = max(50, total_load * 1000 * 0.1 + random.uniform(-10, 10))
            http_mbps = max(0.5, total_load * 0.7)
            ftp_mbps = max(0.1, total_load * 0.2)
        
        # 2. Traffic Monitoring / Feature Extraction
        # Arrival rate scales with total load
        arrival_rate = int(total_load * 120 + random.uniform(-5, 5))
        
        # Base delay increases exponentially as real load approaches arbitrary capacity
        utilization_ratio = min(0.99, total_load / self.capacity_mbps)
        base_delay = 5.0 / (1.0 - utilization_ratio) + random.uniform(-1, 1)
        
        # Queue length builds up when load is high
        queue_length = int(max(0, (total_load - (self.capacity_mbps*0.4)) * 50 + random.uniform(0, 10)))
        
        # 3. ML for Traffic Network (Simulated Logic mapping real values)
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

        if random.random() < 0.2:
            websites = [
                ("Netflix / YouTube 4K", random.uniform(5.0, 25.0)),
                ("Coursera / edX", random.uniform(2.0, 8.0)),
                ("Canvas / Moodle CMS", random.uniform(1.0, 5.0)),
                ("TikTok / Instagram Reels", random.uniform(3.0, 15.0)),
                ("Steam Game Downloads", random.uniform(20.0, 80.0)),
                ("Wikipedia / Web Browsing", random.uniform(0.1, 2.0)),
                ("Zoom / Teams Video call", random.uniform(1.5, 4.0))
            ]
            site, speed = random.choice(websites)
            if state == "low":
                speed *= 1.2
            elif state == "high":
                speed *= 0.3
            alerts.append({"time": now_str, "msg": f"DPI Engine: {site} flow detected consuming {speed:.1f} Mbps", "cls": ""})

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
            self.last_minute = current_min
            
            # Save to Database
            conn = self._get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO network_logs 
                            (time_str, voip_kbps, http_mbps, ftp_mbps, delay_ms, throughput_gbps, packet_loss_pct, state)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            now_str, round(voip_kbps), round(http_mbps, 1), round(ftp_mbps, 1), 
                            round(final_delay, 1), round(final_tput, 2), round(packet_loss, 2), state
                        ))
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting into DB: {e}")
                finally:
                    conn.close()
            
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
    # Fetch from Supabase
    conn = state_manager._get_db_connection()
    if not conn:
        return jsonify({"error": "No database connection"}), 500
        
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    time_str as time, 
                    voip_kbps as voip, 
                    http_mbps as http, 
                    ftp_mbps as ftp,
                    delay_ms as delay, 
                    throughput_gbps as throughput, 
                    packet_loss_pct as loss, 
                    state 
                FROM network_logs 
                ORDER BY timestamp DESC 
                LIMIT 100
            """)
            rows = cur.fetchall()
        return jsonify(rows)
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
