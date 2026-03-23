from flask import Flask, jsonify, send_from_directory, request as flask_request
from flask_cors import CORS
import time
import math
import requests as req_lib
import os
import sqlite3
import psutil
from dotenv import load_dotenv
import joblib
import pandas as pd
import threading
from datetime import datetime, timezone

load_dotenv()  # Load environment variables before initializing classes

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

RADIO_CAPACITY_MBPS = {
    "GSM": 0.2,
    "CDMA": 3.1,
    "UMTS": 42.0,
    "LTE": 150.0,
    "NR": 600.0
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def estimate_tower_traffic(cell, nearby_count):
    """
    OpenCelliD exposes real tower inventory and radio metadata, but not live
    carrier utilization. We derive a tower-aware load estimate from the live
    tower record and return explicit confidence + provenance for the UI.
    """
    radio = str(cell.get('radio') or 'UNKNOWN').upper()
    capacity_mbps = RADIO_CAPACITY_MBPS.get(radio, 50.0)
    samples = max(0.0, float(cell.get('samples') or 0))
    avg_signal = float(cell.get('averageSignal') or cell.get('avgSignal') or -95)
    cell_range_m = max(50.0, float(cell.get('range') or 500))

    sample_factor = clamp(math.log10(samples + 1) / 2.4 if samples else 0.0, 0.0, 1.0)
    signal_factor = clamp((avg_signal + 120.0) / 70.0, 0.15, 1.0)
    density_factor = clamp(nearby_count / 12.0, 0.2, 1.0)
    overlap_factor = clamp(1.0 - (cell_range_m / 4000.0), 0.1, 1.0)

    utilization_pct = clamp(
        8.0
        + (sample_factor * 38.0)
        + (signal_factor * 18.0)
        + (density_factor * 24.0)
        + (overlap_factor * 10.0),
        5.0,
        92.0
    )

    estimated_load_mbps = round(capacity_mbps * (utilization_pct / 100.0), 2)
    confidence = round(
        clamp(
            0.35
            + (0.25 if samples > 0 else 0.0)
            + (0.15 if cell.get('range') else 0.0)
            + (0.10 if cell.get('averageSignal') or cell.get('avgSignal') else 0.0)
            + (0.05 if radio in RADIO_CAPACITY_MBPS else 0.0),
            0.35,
            0.90
        ),
        2
    )

    return {
        "radio_generation": radio,
        "estimated_load_mbps": estimated_load_mbps,
        "estimated_utilization_pct": round(utilization_pct, 1),
        "capacity_mbps": capacity_mbps,
        "traffic_confidence": confidence,
        "traffic_note": "Live carrier tower utilization is not public in OpenCelliD; load is inferred from real tower metadata."
    }


# Simulation State
class NetworkState:
    def __init__(self):
        self.phase = 0.0
        self.last_update = time.time()
        self.config = {
            "voip_alloc": 50,
            "threshold": 0.4,
            "ftp_prio": "std"
        }
        self.last_log_time = 0
        self.log_interval = 10 # Log every 10 seconds
        self.last_state = "low"
        self.on_vercel = os.environ.get('VERCEL', '') == '1'
        self.active_interface = "Vercel Cloud" if self.on_vercel else "Scanning..."
        
        # Real Traffic Baseline
        self.capacity_mbps = 100.0
        self.current_load_mbps = 0.5 if self.on_vercel else 0.01 
        if not self.on_vercel:
            try:
                self.last_net_io = psutil.net_io_counters(pernic=True)
            except Exception:
                self.last_net_io = {}
        else:
            self.last_net_io = {}
        
        # Use /tmp for DB on Vercel (read-only FS); fall back to project dir locally
        import tempfile
        if self.on_vercel:
            self.db_path = os.path.join(tempfile.gettempdir(), "network_logs.db")
        else:
            self.db_path = "network_logs.db"
        self._init_db()
        
        # Load Decision Tree Model
        try:
            self.model = joblib.load('traffic_model.joblib')
            print("ML Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Thread safety for shared state
        self.lock = threading.RLock()

        # Background threads only make sense in a long-running process (not Vercel serverless)
        if not self.on_vercel:
            self.bg_thread = threading.Thread(target=self._background_logger, daemon=True)
            self.bg_thread.start()
            self.traffic_thread = threading.Thread(target=self._traffic_monitor, daemon=True)
            self.traffic_thread.start()
            print("Background Threads (Logger & Traffic) started")
        else:
            print("Vercel environment detected — background threads disabled")

    def _traffic_monitor(self):
        """Continuously samples network IO every 2 seconds to calculate a stable Mbps rate."""
        while True:
            try:
                t1 = time.time()
                io1 = psutil.net_io_counters(pernic=True)
                time.sleep(2)
                t2 = time.time()
                io2 = psutil.net_io_counters(pernic=True)
                
                dt = t2 - t1
                
                # Filter for Wi-Fi or Data (Cellular) interfaces primarily
                sent_bytes = 0
                recv_bytes = 0
                preferred_interfaces = ['wi-fi', 'wifi', 'cellular', 'mobile data', 'ethernet']
                
                # Check for activity on preferred interfaces
                active_nics = []
                for nic, stats1 in io1.items():
                    if any(pref in nic.lower() for pref in preferred_interfaces):
                        if nic in io2:
                            stats2 = io2[nic]
                            s_diff = stats2.bytes_sent - stats1.bytes_sent
                            r_diff = stats2.bytes_recv - stats1.bytes_recv
                            if s_diff > 0 or r_diff > 0:
                                sent_bytes += s_diff
                                recv_bytes += r_diff
                                active_nics.append(nic)
                
                # Fallback: if no hardware nics show activity, check everything else except loopback
                if (sent_bytes + recv_bytes) == 0:
                    for nic, stats1 in io1.items():
                        if 'loopback' not in nic.lower() and 'pseudo' not in nic.lower():
                            if nic in io2:
                                stats2 = io2[nic]
                                s_diff = stats2.bytes_sent - stats1.bytes_sent
                                r_diff = stats2.bytes_recv - stats1.bytes_recv
                                sent_bytes += s_diff
                                recv_bytes += r_diff
                                if (s_diff + r_diff) > 0: active_nics.append(nic)

                with self.lock:
                    self.current_sent_mbps = (sent_bytes * 8) / (dt * 1000000)
                    self.current_recv_mbps = (recv_bytes * 8) / (dt * 1000000)
                    self.current_load_mbps = self.current_sent_mbps + self.current_recv_mbps
                    self.active_interface = active_nics[0] if active_nics else "Auto-Select"
                    # Smoothly decay if 0, but keep at least a tiny baseline
                    self.current_load_mbps = max(0.01, self.current_load_mbps)
                    
            except Exception as e:
                print(f"Error in traffic monitor: {e}")
                time.sleep(5)

    def _background_logger(self):
        """Infinite loop for local environments to log metrics at 10s boundaries."""
        while True:
            try:
                now = time.time()
                sleep_time = self.log_interval - (now % self.log_interval)
                if sleep_time < 0.1: sleep_time += self.log_interval
                time.sleep(sleep_time)
                self._log_to_db()
            except Exception as e:
                print(f"Error in background logger: {e}")
                time.sleep(2)

    def _log_to_db(self):
        """Calculates metrics and inserts into SQLite."""
        try:
            with self.lock:
                now_ts = time.time()
                # Prevent double-logging within the same interval
                if now_ts - self.last_log_time < (self.log_interval - 1):
                    return
                self.last_log_time = now_ts

            metrics = self.get_current_metrics(internal=True)
            if not metrics:
                return

            voip = metrics['traffic']['voip']
            http = metrics['traffic']['http']
            ftp = metrics['traffic']['ftp']
            delay = metrics['performance']['delay']
            tput = metrics['performance']['throughput']
            loss = metrics['performance']['packet_loss']
            state = metrics['ml']['state']
            now = datetime.now(timezone.utc).isoformat()
            now_local = datetime.now()
            now_str = now_local.strftime("%H:%M:%S")

            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO network_logs 
                        (timestamp, time_str, voip_kbps, http_mbps, ftp_mbps, delay_ms, throughput_gbps, packet_loss_pct, state)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        now, now_str, voip, http, ftp, delay, tput, loss, state
                    ))
                    conn.commit()
                except Exception as e:
                    print(f"Error inserting into DB: {e}")
                finally:
                    conn.close()
        except Exception as e:
            print(f"Error in _log_to_db: {e}")

    def _get_db_connection(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

    def _init_db(self):
        conn = self._get_db_connection()
        if not conn: return
        
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS network_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
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
        
    def get_current_metrics(self, internal=False):
        now = time.time()
        import random
        
        # Consistent locking for both internal and external callers
        try:
            with self.lock:
                # 1. Read stable traffic from monitor thread
                # If on Vercel, generate a dynamic sinusoid for the "cloud feel"
                if self.on_vercel:
                    t = time.time()
                    base = 0.5 + 0.3 * math.sin(t / 10.0) + random.uniform(-0.1, 0.1)
                    self.current_load_mbps = max(0.1, base)
                    self.current_sent_mbps = self.current_load_mbps * 0.4
                    self.current_recv_mbps = self.current_load_mbps * 0.6
                
                total_load = self.current_load_mbps
    
                # Distribute real load proportionally into our standard traffic bins for UI
                if total_load < 1.0:
                    voip_kbps = max(20, total_load * 1000 * 0.4)
                    http_mbps = max(0.1, total_load * 0.5)
                    ftp_mbps = max(0.0, total_load * 0.1)
                else:
                    voip_kbps = max(50, total_load * 1000 * 0.1)
                    http_mbps = max(0.5, total_load * 0.7)
                    ftp_mbps = max(0.1, total_load * 0.2)
                
                # 2. Traffic Monitoring / Feature Extraction
                # Arrival rate scales with total load
                arrival_rate = int(total_load * 120 + random.randint(0, 5))
                
                # Base delay increases exponentially as real load approaches arbitrary capacity
                # Lowering virtual capacity to 25 Mbps for high sensitivity to Wi-Fi/Mobile browsing
                virtual_capacity = 25.0
                utilization_ratio = min(0.99, total_load / virtual_capacity) 
                # Add natural jitter even at low load
                jitter = random.uniform(-0.5, 0.5) if total_load > 0 else 0
                base_delay = (5.0 / (1.0 - utilization_ratio)) + jitter
                
                # Active Queue activity (starting at 1 Mbps - typical web browsing)
                if total_load > 0.1:
                    queue_length = 5 + int(max(0, (total_load - 1.0) * 20))
                    queue_length += random.randint(-1, 1)
                else: 
                    queue_length = 0
                queue_length = max(0, queue_length)
                
                # 3. Decision Tree ML for Traffic Network
                start_time = time.perf_counter()
                if self.model:
                    # Prepare features for prediction
                    features = pd.DataFrame([[total_load, base_delay, queue_length, arrival_rate]], 
                                            columns=['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate'])
                    state = self.model.predict(features)[0]
                    
                    # Use probability for confidence if available
                    try:
                        probs = self.model.predict_proba(features)[0]
                        confidence = round(max(probs) * 100, 1)
                    except:
                        confidence = 95.0
                else:
                    # Fallback to threshold logic if model not loaded
                    if utilization_ratio < self.config['threshold']:
                        state = "low"
                    elif utilization_ratio < min(0.95, self.config['threshold'] + 0.35):
                        state = "med"
                    else:
                        state = "high"
                    confidence = 80.0
                    
                infer_time = round((time.perf_counter() - start_time) * 1000, 2)
    
                # 4. Adaptive QoS Controller (Resource Allocation)
                # Logic derived from ML-Driven Adaptive QoS script
                if state == "low":
                    # More balanced priorities (5, 4, 3)
                    bw_voip, bw_http, bw_ftp = 25, 45, 30
                    q_voip, q_http, q_ftp = 10, 30, 60
                elif state == "med":
                    # Moderate VoIP preference (6, 3, 2)
                    bw_voip, bw_http, bw_ftp = 40, 35, 25
                    q_voip, q_http, q_ftp = 20, 40, 40
                    # Apply FTP Priority Logic from User Config
                    if self.config['ftp_prio'] == "high": bw_ftp += 10; bw_http -= 10
                    if self.config['ftp_prio'] == "low": bw_ftp -= 10; bw_http += 10
                else:
                    # Strong VoIP protection (8, 2, 1)
                    bw_voip = max(60, self.config['voip_alloc'])
                    rem = 100 - bw_voip
                    bw_http = int(rem * 0.7)
                    bw_ftp = rem - bw_http
                    q_voip, q_http, q_ftp = 50, 30, 20
    
                # 5. Performance Comparison Engine (Adaptive vs FIFO)
                link_utilization = min(100, int(utilization_ratio * 100))
                queue_occupancy = min(100, int((queue_length / 500.0) * 100))
                
                # Baseline (FIFO) metrics
                fifo_delay = base_delay
                fifo_loss = 0.05 + (utilization_ratio ** 2) * 10.0 if utilization_ratio > 0.4 else 0.01
                fifo_tput = total_load * (1.0 - fifo_loss/100.0)

                # ML-Adaptive metrics
                if state == "high":
                    # Improved protection saves time-sensitive traffic (VoIP)
                    final_delay = base_delay * 0.75  
                    packet_loss = fifo_loss * 0.6
                elif state == "med":
                    final_delay = base_delay * 0.85
                    packet_loss = fifo_loss * 0.4
                else:
                    final_delay = base_delay * 0.95
                    packet_loss = fifo_loss * 0.2
                    
                final_tput = total_load * (1.0 - packet_loss/100.0)

                # Calculate Improvements (%)
                improvement_delay = max(0, ((fifo_delay - final_delay) / fifo_delay) * 100)
                improvement_loss = max(0, ((fifo_loss - packet_loss) / max(0.01, fifo_loss)) * 100)
                improvement_tput = max(0, ((final_tput - fifo_tput) / max(0.01, fifo_tput)) * 100)
    
                # Generate Alerts based on actual state transitions or thresholds
                alerts = []
                now_str = datetime.now().strftime("%H:%M:%S")

                # State Transition Alerts
                if state != self.last_state:
                    if state == "high":
                        alerts.append({"time": now_str, "msg": f"SYSTEM: High Congestion Detected - QoS Policy Active ({bw_voip}% VoIP Reservation)", "cls": "warn"})
                    elif state == "med":
                        alerts.append({"time": now_str, "msg": "SYSTEM: Moderate Load Detected - Adjusting Bandwidth Allocation", "cls": "ok"})
                    elif state == "low":
                        alerts.append({"time": now_str, "msg": "SYSTEM: Nominal Traffic Conditions - Policy Reset to Baseline", "cls": "ok"})
                    self.last_state = state

                # Critical Threshold Alerts (always send if active)
                if utilization_ratio > 0.85:
                    alerts.append({"time": now_str, "msg": "CRITICAL: Link utilization exceeded 85% safety threshold", "cls": "crit"})
                    
                if packet_loss > 1.0:
                     alerts.append({"time": now_str, "msg": f"ALERT: Elevated Packet Loss ({packet_loss:.1f}%) detected", "cls": "warn"})
    
                # Get real active processes causing I/O traffic (local only — not available on Vercel)
                active_processes = []
                if not self.on_vercel:
                    try:
                        procs = []
                        for p in psutil.process_iter(['name', 'io_counters']):
                            try:
                                io = p.info.get('io_counters')
                                if io:
                                    total_io = getattr(io, 'read_bytes', getattr(io, 'read_count', 0)) + \
                                               getattr(io, 'write_bytes', getattr(io, 'write_count', 0))
                                    procs.append((p.info['name'], total_io))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                        sorted_procs = sorted(procs, key=lambda x: x[1], reverse=True)
                        active_processes = [name for name, _ in sorted_procs[:5] if name != 'System Idle Process']
                    except Exception as e:
                        print(f"Error fetching processes: {e}")
    
                if not active_processes:
                    active_processes = ["System Kernel", "Network Interface", "DCN Controller"]
    
                metrics_data = {
                    "processes": active_processes,
                    "traffic": {
                        "voip": round(voip_kbps),
                        "http": round(http_mbps, 1),
                        "ftp": round(ftp_mbps, 1),
                        "aggregate": round(total_load, 1),
                        "sent": round(getattr(self, 'current_sent_mbps', 0), 2),
                        "recv": round(getattr(self, 'current_recv_mbps', 0), 2)
                    },
                    "monitoring": {
                        "arrival_rate": arrival_rate,
                        "delay": round(base_delay, 1),
                        "queue_length": queue_length,
                        "interface": self.active_interface
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
                        "packet_loss": round(packet_loss, 2),
                        "fifo_delay": round(fifo_delay, 1),
                        "fifo_loss": round(fifo_loss, 2),
                        "improvement_delay": round(improvement_delay, 1),
                        "improvement_loss": round(improvement_loss, 1),
                        "improvement_tput": round(improvement_tput, 1)
                    },
                    "alerts": alerts
                }
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            raise
        finally:
            pass

        return metrics_data

state_manager = NetworkState()

@app.route('/api/status', methods=['GET'])
def get_status():
    # Trigger lazy log for serverless environments
    if state_manager.on_vercel:
        state_manager._log_to_db()
    return jsonify(state_manager.get_current_metrics())

@app.route('/api/towers', methods=['GET'])
def get_towers():
    """Return real OpenCelliD towers plus clearly labeled inferred load metrics."""
    lat  = flask_request.args.get('lat', type=float)
    lng  = flask_request.args.get('lng', type=float)
    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng required'}), 400

    OCID_KEY = os.environ.get('OPENCELLID_API_KEY')
    if not OCID_KEY:
        return jsonify({
            'error': 'OPENCELLID_API_KEY is not configured',
            'cells': [],
            'real_traffic_available': False
        }), 503

    delta    = 0.008         # ~800m radius (safely under 4,000,000 sq. mts limit)
    bbox     = f"{lat-delta},{lng-delta},{lat+delta},{lng+delta}"
    url      = f"https://opencellid.org/cell/getInArea?key={OCID_KEY}&BBOX={bbox}&format=json&limit=500"

    try:
        resp = req_lib.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data:
            return jsonify({'error': data['error'], 'cells': []}), 401

        raw_cells = data.get('cells', []) or []
        nearby_count = len(raw_cells)
        enriched_cells = []
        total_estimated_load = 0.0

        for cell in raw_cells:
            enriched_cell = dict(cell)
            traffic = estimate_tower_traffic(enriched_cell, nearby_count)
            enriched_cell.update(traffic)
            total_estimated_load += traffic['estimated_load_mbps']
            enriched_cells.append(enriched_cell)

        avg_utilization = round(
            sum(cell['estimated_utilization_pct'] for cell in enriched_cells) / max(1, len(enriched_cells)),
            1
        )

        data['cells'] = enriched_cells
        data['source'] = 'OpenCelliD'
        data['real_traffic_available'] = False
        data['traffic_mode'] = 'inferred_from_live_tower_inventory'
        data['summary'] = {
            'tower_count': nearby_count,
            'estimated_total_load_mbps': round(total_estimated_load, 2),
            'average_utilization_pct': avg_utilization,
            'note': 'Tower records are live from OpenCelliD. Traffic values are inferred because public carrier APIs do not expose real per-tower utilization.'
        }
        return jsonify(data)
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
    # Fetch from SQLite
    conn = state_manager._get_db_connection()
    if not conn:
        return jsonify({"error": "No database connection"}), 500
        
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                timestamp,
                time_str as time, 
                voip_kbps as voip, 
                http_mbps as http, 
                ftp_mbps as ftp,
                delay_ms as delay, 
                throughput_gbps as throughput, 
                packet_loss_pct as loss, 
                state 
            FROM network_logs 
            ORDER BY id DESC 
            LIMIT 1000
        """)
        rows = [dict(row) for row in cur.fetchall()]
        # Trigger lazy log check on dataset fetch too
        if state_manager.on_vercel:
             state_manager._log_to_db()
        return jsonify(rows)
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/seed', methods=['POST'])
def seed_dataset():
    """Inject 24 hours of realistic sample rows (every minute) for demo purposes."""
    import random, math
    conn = state_manager._get_db_connection()
    if not conn:
        return jsonify({"error": "No DB connection"}), 500
    try:
        cur = conn.cursor()
        rows = []
        base_time = datetime.now(timezone.utc)
        # Generate 144 rows: every 10 minutes over 24 hours (going backwards)
        for i in range(144, 0, -1):
            t = base_time - __import__('datetime').timedelta(minutes=i * 10)
            local_t = t.astimezone()  # convert to local
            time_str = local_t.strftime("%H:%M:%S")
            # Simulate periodic load pattern with sine wave
            phase = (i / 144) * 4 * math.pi
            load = 0.5 + 0.35 * math.sin(phase) + random.uniform(-0.05, 0.05)
            load = max(0.05, min(0.95, load))

            voip = int(max(20, load * 1000 * 0.15))
            http = round(max(0.1, load * 8), 1)
            ftp = round(max(0.0, load * 2), 1)

            if load < 0.35:
                state = "low"
                delay = round(5.0 / (1.0 - load) * random.uniform(0.95, 1.05), 1)
                loss = round(random.uniform(0.01, 0.05), 2)
            elif load < 0.70:
                state = "med"
                delay = round(5.0 / (1.0 - load) * random.uniform(0.88, 1.10), 1)
                loss = round(random.uniform(0.05, 0.3), 2)
            else:
                state = "high"
                delay = round(5.0 / (1.0 - load) * random.uniform(0.80, 1.15), 1)
                loss = round(0.5 + (load - 0.70) * 5.0 + random.uniform(-0.1, 0.2), 2)

            tput = round(load * 100 * (1.0 - loss / 100.0), 2)
            rows.append((t.isoformat(), time_str, voip, http, ftp, delay, tput, loss, state))

        cur.executemany("""
            INSERT INTO network_logs
            (timestamp, time_str, voip_kbps, http_mbps, ftp_mbps, delay_ms, throughput_gbps, packet_loss_pct, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        return jsonify({"status": "ok", "inserted": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/api/dataset/clear', methods=['POST'])
def clear_dataset():
    """Wipe all rows from network_logs (for dev/demo reset)."""
    conn = state_manager._get_db_connection()
    if not conn:
        return jsonify({"error": "No DB connection"}), 500
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM network_logs")
        conn.commit()
        return jsonify({"status": "ok", "deleted": cur.rowcount})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=False, host='0.0.0.0', port=port)
