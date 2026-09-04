import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def send_one(session, url, image_bytes):
    files = {"file": ("test.png", image_bytes, "image/png")}
    start = time.time()
    try:
        r = session.post(url + "/predict", files=files, timeout=10)
        latency = int((time.time() - start) * 1000)
        return latency, r.status_code, r.status_code == 200
    except Exception:
        latency = int((time.time() - start) * 1000)
        return latency, 0, False


def percentile(sorted_list, p):
    if not sorted_list:
        return 0
    k = int((p / 100.0) * len(sorted_list)) - 1
    k = max(0, min(k, len(sorted_list) - 1))
    return sorted_list[k]


def main():
    if len(sys.argv) < 3:
        print("Usage: python python_load_test.py <baseUrl> <imagePath> [numRequests] [concurrency]")
        sys.exit(2)

    base = sys.argv[1]
    img_path = sys.argv[2]
    num_requests = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    with open(img_path, "rb") as f:
        img = f.read()

    print(f"Target: {base}/predict | requests={num_requests} | concurrency={concurrency}")
    start_wall = time.time()
    results = []
    errors = 0
    session = requests.Session()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send_one, session, base, img) for _ in range(num_requests)]
        for fut in as_completed(futures):
            lat, code, ok = fut.result()
            results.append((lat, code, ok))
            if not ok:
                errors += 1

    wall_ms = int((time.time() - start_wall) * 1000)

    latencies = sorted([r[0] for r in results if r[2]])
    total = len(results) + errors
    success = len([1 for r in results if r[2]])
    throughput = total / (wall_ms / 1000.0) if wall_ms > 0 else 0

    print("\n=== Load Test Report ===")
    print(f"Total requests:   {total}")
    print(f"Successful:       {success}")
    print(f"Errors:           {errors}")
    print(f"Wall time:        {wall_ms} ms")
    print(f"Throughput:       {throughput:.1f} req/s")
    if latencies:
        print(f"Latency p50:      {percentile(latencies, 50)} ms")
        print(f"Latency p90:      {percentile(latencies, 90)} ms")
        print(f"Latency p95:      {percentile(latencies, 95)} ms")
        print(f"Latency p99:      {percentile(latencies, 99)} ms")
        print(f"Latency max:      {latencies[-1]} ms")


if __name__ == '__main__':
    main()
