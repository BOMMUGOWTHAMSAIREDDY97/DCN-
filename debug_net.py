import psutil
import time

def list_interfaces():
    print("Listing network interfaces and activity (2s sample)...")
    io1 = psutil.net_io_counters(pernic=True)
    time.sleep(2)
    io2 = psutil.net_io_counters(pernic=True)
    
    print(f"{'Interface':<30} {'Sent (KB/s)':<15} {'Recv (KB/s)':<15}")
    print("-" * 60)
    for nic, stats1 in io1.items():
        if nic in io2:
            stats2 = io2[nic]
            sent = (stats2.bytes_sent - stats1.bytes_sent) / 1024 / 2
            recv = (stats2.bytes_recv - stats1.bytes_recv) / 1024 / 2
            if sent > 0 or recv > 0:
                print(f"{nic:<30} {sent:<15.2f} {recv:<15.2f}")

if __name__ == "__main__":
    list_interfaces()
