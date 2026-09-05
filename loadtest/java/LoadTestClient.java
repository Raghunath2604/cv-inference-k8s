/**
 * Concurrent synthetic-monitoring / load-test client for the
 * /predict endpoint, written against the JDK's built-in java.net.http
 * client only — no external dependencies (no Maven/Gradle build
 * needed), so it compiles and runs anywhere a JDK 11+ is installed
 * with a single `javac` call.
 *
 * Usage:
 *   javac LoadTestClient.java
 *   java LoadTestClient http://localhost:8000 <path/to/test-image.png> [numRequests] [concurrency]
 *
 * Uses Java 21 virtual threads to cheaply model high concurrency.
 */
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class LoadTestClient {

    record Result(long latencyMs, int statusCode, boolean ok) {}

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java LoadTestClient <baseUrl> <imagePath> [numRequests] [concurrency]");
            System.exit(2);
        }
        String baseUrl = args[0];
        Path imagePath = Path.of(args[1]);
        int numRequests = args.length > 2 ? Integer.parseInt(args[2]) : 200;
        int concurrency = args.length > 3 ? Integer.parseInt(args[3]) : 20;

        byte[] imageBytes = Files.readAllBytes(imagePath);
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)  // uvicorn is HTTP/1.1-only;
                // HttpClient defaults to preferring HTTP/2 with a cleartext
                // upgrade attempt, which uvicorn's parser rejects outright
                // ("Invalid HTTP request received.") rather than falling
                // back cleanly — found the hard way building this the
                // first time, forcing HTTP/1.1 explicitly avoids it.
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        List<Result> results = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger errors = new AtomicInteger();

        System.out.printf("Target: %s/predict | requests=%d | concurrency=%d%n",
                baseUrl, numRequests, concurrency);

        long wallStart = System.nanoTime();
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            Semaphore inFlight = new Semaphore(concurrency);
            List<Future<?>> futures = new ArrayList<>();

            for (int i = 0; i < numRequests; i++) {
                inFlight.acquire();
                futures.add(pool.submit(() -> {
                    try {
                        results.add(sendOne(client, baseUrl, imageBytes));
                    } catch (Exception e) {
                        errors.incrementAndGet();
                    } finally {
                        inFlight.release();
                    }
                }));
            }
            for (Future<?> f : futures) f.get();
        }
        long wallMs = (System.nanoTime() - wallStart) / 1_000_000;

        printReport(results, errors.get(), wallMs);
    }

    private static Result sendOne(HttpClient client, String baseUrl, byte[] imageBytes) throws IOException, InterruptedException {
        String boundary = "----javaLoadTest" + System.nanoTime();
        byte[] body = buildMultipartBody(boundary, imageBytes);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/predict"))
                .timeout(Duration.ofSeconds(10))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(BodyPublishers.ofByteArray(body))
                .build();

        long start = System.nanoTime();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        long latencyMs = (System.nanoTime() - start) / 1_000_000;

        return new Result(latencyMs, response.statusCode(), response.statusCode() == 200);
    }

    private static byte[] buildMultipartBody(String boundary, byte[] imageBytes) throws IOException {
        var out = new java.io.ByteArrayOutputStream();
        String header = "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n"
                + "Content-Type: image/png\r\n\r\n";
        out.write(header.getBytes());
        out.write(imageBytes);
        out.write(("\r\n--" + boundary + "--\r\n").getBytes());
        return out.toByteArray();
    }

    private static void printReport(List<Result> results, int errors, long wallMs) {
        List<Long> latencies = new ArrayList<>();
        int successCount = 0;
        for (Result r : results) {
            if (r.ok()) {
                latencies.add(r.latencyMs());
                successCount++;
            }
        }
        latencies.sort(Long::compareTo);

        int total = results.size() + errors;
        double throughput = total / (wallMs / 1000.0);

        System.out.println("\n=== Load Test Report ===");
        System.out.printf("Total requests:   %d%n", total);
        System.out.printf("Successful:       %d%n", successCount);
        System.out.printf("Errors:           %d%n", errors);
        System.out.printf("Wall time:        %d ms%n", wallMs);
        System.out.printf("Throughput:       %.1f req/s%n", throughput);

        if (!latencies.isEmpty()) {
            System.out.printf("Latency p50:      %d ms%n", percentile(latencies, 50));
            System.out.printf("Latency p90:      %d ms%n", percentile(latencies, 90));
            System.out.printf("Latency p95:      %d ms%n", percentile(latencies, 95));
            System.out.printf("Latency p99:      %d ms%n", percentile(latencies, 99));
            System.out.printf("Latency max:      %d ms%n", latencies.get(latencies.size() - 1));
        }
    }

    private static long percentile(List<Long> sortedLatencies, int p) {
        int idx = (int) Math.ceil(p / 100.0 * sortedLatencies.size()) - 1;
        idx = Math.max(0, Math.min(idx, sortedLatencies.size() - 1));
        return sortedLatencies.get(idx);
    }
}
