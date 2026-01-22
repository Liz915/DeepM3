# =========================
# System Efficiency Simulation
# =========================

NUM_REQUESTS = 1000

# Assumptions
L2_LAT = 10       # ms
L3_LAT = 2000     # ms

L2_COST = 0.001   # $
L3_COST = 0.005   # $

SIMPLE_RATIO = 0.8
COMPLEX_RATIO = 0.2

def all_l3():
    latency = NUM_REQUESTS * L3_LAT
    cost = NUM_REQUESTS * L3_COST
    return latency / NUM_REQUESTS, cost

def reason_m3():
    l2_req = int(NUM_REQUESTS * SIMPLE_RATIO)
    l3_req = int(NUM_REQUESTS * COMPLEX_RATIO)

    latency = (l2_req * L2_LAT + l3_req * L3_LAT) / NUM_REQUESTS
    cost = l2_req * L2_COST + l3_req * L3_COST
    return latency, cost

if __name__ == "__main__":
    lat_all, cost_all = all_l3()
    lat_m3, cost_m3 = reason_m3()

    print("\n===== System Efficiency =====\n")
    print(f"{'Method':<15} {'Avg Latency(ms)':<20} {'Total Cost($)':<15}")
    print("-" * 50)
    print(f"{'All-L3':<15} {lat_all:<20.2f} {cost_all:<15.2f}")
    print(f"{'Reason-M3':<15} {lat_m3:<20.2f} {cost_m3:<15.2f}")

    print("\nLatency Speedup: {:.1f}x".format(lat_all / lat_m3))
    print("Cost Reduction: {:.0f}%".format((1 - cost_m3 / cost_all) * 100))