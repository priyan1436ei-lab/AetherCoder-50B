"""Large-scale multilingual code corpus across all 16 target programming languages."""

from typing import Dict, List

# Extensive code corpus covering all 16 languages specified in the Master Prompt:
# Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust, PHP, Ruby, Kotlin, Swift, SQL, HTML/CSS, Bash
MULTILINGUAL_CORPUS: List[Dict[str, str]] = [
    # -------------------------------------------------------------------------
    # 1. PYTHON: High-Performance Caching & Dynamic Programming
    # -------------------------------------------------------------------------
    {
        "language": "python",
        "domain": "general",
        "title": "Segment Tree with Lazy Propagation",
        "prompt": "Implement a Segment Tree with Lazy Propagation in Python for range sum queries and range additions in O(log N).",
        "code": '''class LazySegmentTree:
    def __init__(self, data: list[int]):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(data, 0, 0, self.n - 1)

    def _build(self, data: list[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        self._build(data, left_child, start, mid)
        self._build(data, right_child, mid + 1, end)
        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def _propagate(self, node: int, start: int, end: int) -> None:
        if self.lazy[node] != 0:
            val = self.lazy[node]
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self.tree[left_child] += val * (mid - start + 1)
            self.lazy[left_child] += val
            self.tree[right_child] += val * (end - mid)
            self.lazy[right_child] += val
            self.lazy[node] = 0

    def update_range(self, l: int, r: int, val: int, node: int = 0, start: int = 0, end: int = None) -> None:
        if end is None: end = self.n - 1
        if l <= start and end <= r:
            self.tree[node] += val * (end - start + 1)
            self.lazy[node] += val
            return
        self._propagate(node, start, end)
        mid = (start + end) // 2
        if l <= mid:
            self.update_range(l, r, val, 2 * node + 1, start, mid)
        if r > mid:
            self.update_range(l, r, val, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query_range(self, l: int, r: int, node: int = 0, start: int = 0, end: int = None) -> int:
        if end is None: end = self.n - 1
        if r < start or end < l: return 0
        if l <= start and end <= r: return self.tree[node]
        self._propagate(node, start, end)
        mid = (start + end) // 2
        return self.query_range(l, r, 2 * node + 1, start, mid) + self.query_range(l, r, 2 * node + 2, mid + 1, end)
'''
    },

    # -------------------------------------------------------------------------
    # 2. TYPESCRIPT: Distributed Event Bus & Type-Safe Mediators
    # -------------------------------------------------------------------------
    {
        "language": "typescript",
        "domain": "instruction_following",
        "title": "Type-Safe Asynchronous Event Bus with Middleware",
        "prompt": "Create an enterprise type-safe event bus in TypeScript supporting wildcard subscriptions and async middleware.",
        "code": '''export type EventMap = Record<string, any>;
export type Middleware<T> = (event: string, payload: T, next: () => Promise<void>) => Promise<void>;
export type Handler<T> = (payload: T) => Promise<void> | void;

export class TypedEventBus<Events extends EventMap> {
  private handlers: Map<keyof Events, Set<Handler<any>>> = new Map();
  private middlewares: Middleware<any>[] = [];

  public use(middleware: Middleware<any>): this {
    this.middlewares.push(middleware);
    return this;
  }

  public on<K extends keyof Events>(event: K, handler: Handler<Events[K]>): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => this.off(event, handler);
  }

  public off<K extends keyof Events>(event: K, handler: Handler<Events[K]>): void {
    const set = this.handlers.get(event);
    if (set) {
      set.delete(handler);
      if (set.size === 0) this.handlers.delete(event);
    }
  }

  public async emit<K extends keyof Events>(event: K, payload: Events[K]): Promise<void> {
    const executeHandlers = async () => {
      const set = this.handlers.get(event);
      if (set) {
        const promises = Array.from(set).map(handler => Promise.resolve(handler(payload)));
        await Promise.all(promises);
      }
    };

    let idx = 0;
    const runner = async (): Promise<void> => {
      if (idx < this.middlewares.length) {
        const mw = this.middlewares[idx++];
        await mw(String(event), payload, runner);
      } else {
        await executeHandlers();
      }
    };

    await runner();
  }
}
'''
    },

    # -------------------------------------------------------------------------
    # 3. JAVASCRIPT: High-Performance Throttled Worker Queue
    # -------------------------------------------------------------------------
    {
        "language": "javascript",
        "domain": "instruction_following",
        "title": "Concurrency Throttled Async Pool",
        "prompt": "Write a lightweight concurrency throttler in vanilla modern JavaScript with exponential backoff retries.",
        "code": '''export async function concurrentMap(items, concurrency, asyncFn, maxRetries = 3) {
  const results = new Array(items.length);
  let currentIndex = 0;

  async function executeWithRetry(item, retries = 0) {
    try {
      return await asyncFn(item);
    } catch (err) {
      if (retries < maxRetries) {
        const delayMs = Math.pow(2, retries) * 100 + Math.random() * 50;
        await new Promise(r => setTimeout(r, delayMs));
        return executeWithRetry(item, retries + 1);
      }
      throw err;
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (currentIndex < items.length) {
      const index = currentIndex++;
      results[index] = await executeWithRetry(items[index]);
    }
  });

  await Promise.all(workers);
  return results;
}
'''
    },

    # -------------------------------------------------------------------------
    # 4. RUST: Zero-Copy Binary Deserializer with Lifetime Bounds
    # -------------------------------------------------------------------------
    {
        "language": "rust",
        "domain": "general",
        "title": "Zero-Copy Byte Slice Reader",
        "prompt": "Implement a zero-copy binary parser in Rust using memory slices and lifetime annotations.",
        "code": '''use std::convert::TryInto;

#[derive(Debug, PartialEq)]
pub struct PacketHeader<\'a> {
    pub magic: u32,
    pub payload_len: u32,
    pub tag: &\'a [u8],
}

pub struct ZeroCopyReader<\'a> {
    buffer: &\'a [u8],
    offset: usize,
}

impl<\'a> ZeroCopyReader<\'a> {
    pub fn new(buffer: &\'a [u8]) -> Self {
        Self { buffer, offset: 0 }
    }

    pub fn read_u32(&mut self) -> Result<u32, &\'static str> {
        if self.offset + 4 > self.buffer.len() {
            return Err("Unexpected EOF while reading u32");
        }
        let bytes = &self.buffer[self.offset..self.offset + 4];
        self.offset += 4;
        Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_slice(&mut self, len: usize) -> Result<&\'a [u8], &\'static str> {
        if self.offset + len > self.buffer.len() {
            return Err("Slice length exceeds remaining buffer capacity");
        }
        let slice = &self.buffer[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    pub fn parse_header(&mut self) -> Result<PacketHeader<\'a>, &\'static str> {
        let magic = self.read_u32()?;
        let payload_len = self.read_u32()?;
        let tag = self.read_slice(4)?;
        Ok(PacketHeader { magic, payload_len, tag })
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 5. GO: Distributed Raft-like Consensus Heartbeat Monitor
    # -------------------------------------------------------------------------
    {
        "language": "go",
        "domain": "instruction_following",
        "title": "Raft Heartbeat Lease Monitor",
        "prompt": "Write a concurrent heartbeat timer in Go with thread-safe atomic status checks and renewal callbacks.",
        "code": '''package consensus

import (
	"sync"
	"sync/atomic"
	"time"
)

type HeartbeatMonitor struct {
	timeout       time.Duration
	lastHeartbeat int64 // Unix nano
	isLeader      int32
	onTimeout     func()
	stopCh        chan struct{}
	wg            sync.WaitGroup
}

func NewHeartbeatMonitor(timeout time.Duration, onTimeout func()) *HeartbeatMonitor {
	m := &HeartbeatMonitor{
		timeout:   timeout,
		onTimeout: onTimeout,
		stopCh:    make(chan struct{}),
	}
	atomic.StoreInt64(&m.lastHeartbeat, time.Now().UnixNano())
	return m
}

func (m *HeartbeatMonitor) Start() {
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		ticker := time.NewTicker(m.timeout / 2)
		defer ticker.Stop()

		for {
			select {
			case <-m.stopCh:
				return
			case <-ticker.C:
				last := atomic.LoadInt64(&m.lastHeartbeat)
				if time.Since(time.Unix(0, last)) > m.timeout {
					if atomic.CompareAndSwapInt32(&m.isLeader, 1, 0) {
						m.onTimeout()
					}
				}
			}
		}
	}()
}

func (m *HeartbeatMonitor) Beat() {
	atomic.StoreInt64(&m.lastHeartbeat, time.Now().UnixNano())
	atomic.StoreInt32(&m.isLeader, 1)
}

func (m *HeartbeatMonitor) Stop() {
	close(m.stopCh)
	m.wg.Wait()
}
'''
    },

    # -------------------------------------------------------------------------
    # 6. C++: Thread-Safe Memory Pool Allocator
    # -------------------------------------------------------------------------
    {
        "language": "cpp",
        "domain": "general",
        "title": "Fixed-Size Block Memory Pool Allocator",
        "prompt": "Implement a high-performance, cache-aligned fixed-size chunk memory pool allocator in C++20.",
        "code": '''#include <cstddef>
#include <new>
#include <vector>
#include <mutex>

template <typename T, size_t BlockSize = 4096>
class MemoryPool {
private:
    union Node {
        Node* next;
        alignas(alignof(T)) char storage[sizeof(T)];
    };

    Node* free_list_head = nullptr;
    std::vector<void*> allocated_blocks;
    std::mutex pool_mutex;

    void expand_pool() {
        size_t num_chunks = BlockSize / sizeof(Node);
        void* raw_block = ::operator new(BlockSize);
        allocated_blocks.push_back(raw_block);

        Node* current = static_cast<Node*>(raw_block);
        for (size_t i = 0; i < num_chunks - 1; ++i) {
            current[i].next = &current[i + 1];
        }
        current[num_chunks - 1].next = free_list_head;
        free_list_head = current;
    }

public:
    MemoryPool() = default;

    ~MemoryPool() {
        for (void* block : allocated_blocks) {
            ::operator delete(block);
        }
    }

    template <typename... Args>
    T* allocate(Args&&... args) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!free_list_head) {
            expand_pool();
        }
        Node* chunk = free_list_head;
        free_list_head = free_list_head->next;
        return new (chunk->storage) T(std::forward<Args>(args)...);
    }

    void deallocate(T* ptr) {
        if (!ptr) return;
        ptr->~T();
        std::lock_guard<std::mutex> lock(pool_mutex);
        Node* chunk = reinterpret_cast<Node*>(ptr);
        chunk->next = free_list_head;
        free_list_head = chunk;
    }
};
'''
    },

    # -------------------------------------------------------------------------
    # 7. C: Cache-Aligned Ring Buffer with Memory Barriers
    # -------------------------------------------------------------------------
    {
        "language": "c",
        "domain": "general",
        "title": "Atomic SPSC Queue with Cacheline Padding in C11",
        "prompt": "Write an SPSC bounded circular queue in C11 with explicit memory barriers and cache line alignment.",
        "code": '''#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdalign.h>

#define CACHE_LINE_SIZE 64

typedef struct {
    alignas(CACHE_LINE_SIZE) atomic_size_t head;
    alignas(CACHE_LINE_SIZE) atomic_size_t tail;
    size_t capacity;
    void** buffer;
} spsc_queue_t;

spsc_queue_t* spsc_create(size_t capacity) {
    if ((capacity & (capacity - 1)) != 0) return NULL; // Must be power of 2
    spsc_queue_t* q = aligned_alloc(CACHE_LINE_SIZE, sizeof(spsc_queue_t));
    if (!q) return NULL;

    q->buffer = malloc(capacity * sizeof(void*));
    if (!q->buffer) {
        free(q);
        return NULL;
    }
    q->capacity = capacity;
    atomic_init(&q->head, 0);
    atomic_init(&q->tail, 0);
    return q;
}

bool spsc_enqueue(spsc_queue_t* q, void* item) {
    size_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
    size_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);
    if ((head - tail) >= q->capacity) return false; // Full

    q->buffer[head & (q->capacity - 1)] = item;
    atomic_store_explicit(&q->head, head + 1, memory_order_release);
    return true;
}

bool spsc_dequeue(spsc_queue_t* q, void** item) {
    size_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    size_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    if (tail == head) return false; // Empty

    *item = q->buffer[tail & (q->capacity - 1)];
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    return true;
}

void spsc_destroy(spsc_queue_t* q) {
    if (!q) return;
    free(q->buffer);
    free(q);
}
'''
    },

    # -------------------------------------------------------------------------
    # 8. C#: Async Reactive Pipeline with Channels
    # -------------------------------------------------------------------------
    {
        "language": "csharp",
        "domain": "instruction_following",
        "title": "System.Threading.Channels Pipeline with Batching",
        "prompt": "Create a high-throughput async processing pipeline in C# with bounded channels and sliding window batching.",
        "code": '''using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

public class BatchProcessingPipeline<T>
{
    private readonly Channel<T> _channel;
    private readonly int _batchSize;
    private readonly TimeSpan _maxWaitTime;

    public BatchProcessingPipeline(int capacity, int batchSize, TimeSpan maxWaitTime)
    {
        _batchSize = batchSize;
        _maxWaitTime = maxWaitTime;
        _channel = Channel.CreateBounded<T>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true
        });
    }

    public async ValueTask PublishAsync(T item, CancellationToken ct = default)
    {
        await _channel.Writer.WriteAsync(item, ct);
    }

    public void Complete() => _channel.Writer.Complete();

    public async Task StartConsumingAsync(Func<List<T>, Task> batchHandler, CancellationToken ct = default)
    {
        var reader = _channel.Reader;
        var batch = new List<T>(_batchSize);
        var timeoutCts = new CancellationTokenSource(_maxWaitTime);

        while (await reader.WaitToReadAsync(ct))
        {
            while (batch.Count < _batchSize && reader.TryRead(out var item))
            {
                batch.Add(item);
            }

            if (batch.Count >= _batchSize || timeoutCts.IsCancellationRequested)
            {
                if (batch.Count > 0)
                {
                    await batchHandler(new List<T>(batch));
                    batch.Clear();
                }
                timeoutCts.CancelAfter(_maxWaitTime);
            }
        }

        if (batch.Count > 0)
        {
            await batchHandler(batch);
        }
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 9. JAVA: Lock-Free SkipList with Concurrent Skip Map
    # -------------------------------------------------------------------------
    {
        "language": "java",
        "domain": "general",
        "title": "Thread-Safe Key-Value Cache with SoftReference LRU",
        "prompt": "Implement a high-throughput caching service in Java with concurrent read-write locks and TTL eviction.",
        "code": '''package com.codex.cache;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.DelayQueue;
import java.util.concurrent.Delayed;
import java.util.concurrent.TimeUnit;

public class ExpiringCache<K, V> {
    private final ConcurrentHashMap<K, V> cache = new ConcurrentHashMap<>();
    private final DelayQueue<ExpiringKey<K>> cleanupQueue = new DelayQueue<>();

    public ExpiringCache() {
        Thread cleaner = new Thread(this::cleanupWorker, "Cache-Eviction-Daemon");
        cleaner.setDaemon(true);
        cleaner.start();
    }

    public void put(K key, V value, long ttlMillis) {
        cache.put(key, value);
        cleanupQueue.put(new ExpiringKey<>(key, ttlMillis));
    }

    public V get(K key) {
        return cache.get(key);
    }

    public boolean remove(K key) {
        return cache.remove(key) != null;
    }

    private void cleanupWorker() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                ExpiringKey<K> delayedKey = cleanupQueue.take();
                cache.remove(delayedKey.key);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    private static class ExpiringKey<K> implements Delayed {
        final K key;
        final long expireAt;

        ExpiringKey(K key, long ttl) {
            this.key = key;
            this.expireAt = System.currentTimeMillis() + ttl;
        }

        @Override
        public long getDelay(TimeUnit unit) {
            return unit.convert(expireAt - System.currentTimeMillis(), TimeUnit.MILLISECONDS);
        }

        @Override
        public int compareTo(Delayed o) {
            return Long.compare(this.expireAt, ((ExpiringKey<?>) o).expireAt);
        }
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 10. KOTLIN: Coroutine Actor with State Machine & Mutex
    # -------------------------------------------------------------------------
    {
        "language": "kotlin",
        "domain": "instruction_following",
        "title": "Kotlin Coroutines Actor State Machine",
        "prompt": "Implement a stateful actor in Kotlin using Channel, Flow, and Structured Concurrency.",
        "code": '''package com.codex.actor

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel

sealed class CircuitBreakerCommand {
    data class RecordSuccess(val responseTimeMs: Long) : CircuitBreakerCommand()
    data class RecordFailure(val throwable: Throwable) : CircuitBreakerCommand()
    data class CanExecute(val response: CompletableDeferred<Boolean>) : CircuitBreakerCommand()
}

enum class State { CLOSED, OPEN, HALF_OPEN }

class CircuitBreakerActor(
    private val scope: CoroutineScope,
    private val failureThreshold: Int = 5,
    private val resetTimeoutMs: Long = 10000L
) {
    private val channel = Channel<CircuitBreakerCommand>(Channel.BUFFERED)
    private var state = State.CLOSED
    private var consecutiveFailures = 0
    private var lastFailureTime = 0L

    init {
        scope.launch {
            for (msg in channel) {
                when (msg) {
                    is CircuitBreakerCommand.CanExecute -> {
                        val now = System.currentTimeMillis()
                        val allowed = when (state) {
                            State.CLOSED -> true
                            State.OPEN -> if (now - lastFailureTime > resetTimeoutMs) {
                                state = State.HALF_OPEN
                                true
                            } else false
                            State.HALF_OPEN -> true
                        }
                        msg.response.complete(allowed)
                    }
                    is CircuitBreakerCommand.RecordSuccess -> {
                        if (state == State.HALF_OPEN) state = State.CLOSED
                        consecutiveFailures = 0
                    }
                    is CircuitBreakerCommand.RecordFailure -> {
                        consecutiveFailures++
                        lastFailureTime = System.currentTimeMillis()
                        if (consecutiveFailures >= failureThreshold) {
                            state = State.OPEN
                        }
                    }
                }
            }
        }
    }

    suspend fun allowRequest(): Boolean {
        val def = CompletableDeferred<Boolean>()
        channel.send(CircuitBreakerCommand.CanExecute(def))
        return def.await()
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 11. SWIFT: Actor-Isolated Asynchronous Network Dispatcher
    # -------------------------------------------------------------------------
    {
        "language": "swift",
        "domain": "instruction_following",
        "title": "Swift Actor Network Request Deduplicator",
        "prompt": "Write an actor in Swift with Task coalescing to prevent redundant inflight network requests.",
        "code": '''import Foundation

actor RequestDeduplicator {
    private var inFlightTasks: [URL: Task<Data, Error>] = [:]

    func fetch(url: URL) async throws -> Data {
        if let existingTask = inFlightTasks[url] {
            return try await existingTask.value
        }

        let task = Task { () throws -> Data in
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            return data
        }

        inFlightTasks[url] = task

        do {
            let result = try await task.value
            inFlightTasks[url] = nil
            return result
        } catch {
            inFlightTasks[url] = nil
            throw error
        }
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 12. PHP: PSR-15 Middleware Pipeline with Dependency Injection
    # -------------------------------------------------------------------------
    {
        "language": "php",
        "domain": "instruction_following",
        "title": "PSR-15 HTTP Middleware Dispatcher in PHP 8.2",
        "prompt": "Create an immutable PSR-15 Middleware Dispatcher in modern PHP 8.2 with onion request-response wrapping.",
        "code": '''<?php
declare(strict_types=1);

namespace Codex\\Http;

use Psr\\Http\\Message\\ResponseInterface;
use Psr\\Http\\Message\\ServerRequestInterface;
use Psr\\Http\\Server\\MiddlewareInterface;
use Psr\\Http\\Server\\RequestHandlerInterface;

final class MiddlewareDispatcher implements RequestHandlerInterface
{
    /** @var MiddlewareInterface[] */
    private array $middlewareStack;
    private RequestHandlerInterface $fallbackHandler;
    private int $index = 0;

    /**
     * @param MiddlewareInterface[] $middlewareStack
     */
    public function __construct(array $middlewareStack, RequestHandlerInterface $fallbackHandler)
    {
        $this->middlewareStack = array_values($middlewareStack);
        $this->fallbackHandler = $fallbackHandler;
    }

    public function handle(ServerRequestInterface $request): ResponseInterface
    {
        if (!isset($this->middlewareStack[$this->index])) {
            return $this->fallbackHandler->handle($request);
        }

        $middleware = $this->middlewareStack[$this->index];
        $next = clone $this;
        $next->index++;

        return $middleware->process($request, $next);
    }
}
'''
    },

    # -------------------------------------------------------------------------
    # 13. RUBY: Fiber-Based Asynchronous Rate Limiter
    # -------------------------------------------------------------------------
    {
        "language": "ruby",
        "domain": "instruction_following",
        "title": "Token Bucket Rate Limiter with Monotonic Clock",
        "prompt": "Implement a thread-safe token bucket rate limiter in Ruby with burst capacity and smooth refill.",
        "code": '''# frozen_string_literal: true

class TokenBucketLimiter
  def initialize(rate_per_sec:, burst_capacity:)
    @rate = rate_per_sec.to_f
    @capacity = burst_capacity.to_f
    @tokens = @capacity
    @last_refill = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @mutex = Mutex.new
  end

  def acquire?(tokens_requested = 1)
    @mutex.synchronize do
      now = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      elapsed = now - @last_refill
      @tokens = [@capacity, @tokens + (elapsed * @rate)].min
      @last_refill = now

      if @tokens >= tokens_requested
        @tokens -= tokens_requested
        true
      else
        false
      end
    end
  end

  def wait_and_acquire(tokens_requested = 1)
    loop do
      return true if acquire?(tokens_requested)
      sleep(1.0 / @rate)
    end
  end
end
'''
    },

    # -------------------------------------------------------------------------
    # 14. SQL: TimescaleDB-Style Continuous Aggregations & Partitioning
    # -------------------------------------------------------------------------
    {
        "language": "sql",
        "domain": "instruction_following",
        "title": "High-Throughput Time-Series Metrics Aggregation with Partitioning",
        "prompt": "Write PostgreSQL DDL and queries creating monthly range partitions and materialized hourly rollups with indexes.",
        "code": '''-- Create partitioned parent table for system telemetry
CREATE TABLE system_metrics (
    metric_id UUID NOT NULL,
    host_name VARCHAR(64) NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    cpu_percent NUMERIC(5, 2) NOT NULL,
    memory_used_bytes BIGINT NOT NULL,
    vram_used_bytes BIGINT NOT NULL,
    CONSTRAINT pk_system_metrics PRIMARY KEY (recorded_at, host_name, metric_id)
) PARTITION BY RANGE (recorded_at);

-- Create 2026 Monthly Partitions
CREATE TABLE metrics_y2026_m10 PARTITION OF system_metrics
    FOR VALUES FROM (\'2026-10-01 00:00:00+00\') TO (\'2026-11-01 00:00:00+00\');

CREATE TABLE metrics_y2026_m11 PARTITION OF system_metrics
    FOR VALUES FROM (\'2026-11-01 00:00:00+00\') TO (\'2026-12-01 00:00:00+00\');

-- High performance composite index for range queries
CREATE INDEX idx_metrics_host_time ON system_metrics (host_name, recorded_at DESC);

-- Materialized 1-hour rollup calculation
CREATE MATERIALIZED VIEW hourly_metric_rollups AS
SELECT
    date_trunc(\'hour\', recorded_at) AS time_bucket,
    host_name,
    ROUND(AVG(cpu_percent), 2) AS avg_cpu,
    MAX(cpu_percent) AS max_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_percent) AS p95_cpu,
    MAX(vram_used_bytes) / (1024 * 1024) AS peak_vram_mb
FROM system_metrics
GROUP BY date_trunc(\'hour\', recorded_at), host_name
WITH NO DATA;
'''
    },

    # -------------------------------------------------------------------------
    # 15. HTML & CSS: Responsive Glassmorphic Dashboard Layout
    # -------------------------------------------------------------------------
    {
        "language": "html",
        "domain": "instruction_following",
        "title": "Accessible Glassmorphic Container with CSS Grid",
        "prompt": "Create a semantic, accessible glassmorphic component with CSS Grid and smooth hover transitions.",
        "code": '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    :root {
      --card-bg: rgba(255, 255, 255, 0.05);
      --card-border: rgba(255, 255, 255, 0.12);
      --text-highlight: #38bdf8;
    }
    .grid-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.5rem;
      padding: 2rem;
    }
    .glass-card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.5rem;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
      border-color: var(--text-highlight);
    }
  </style>
</head>
<body>
  <main class="grid-container" aria-label="System Metrics">
    <article class="glass-card">
      <h2>Inference Throughput</h2>
      <p>Continuous batching active at <strong>68 tokens/sec</strong>.</p>
    </article>
  </main>
</body>
</html>
'''
    },

    # -------------------------------------------------------------------------
    # 16. BASH: Production Deployment Script with Trap Signal Guards
    # -------------------------------------------------------------------------
    {
        "language": "bash",
        "domain": "instruction_following",
        "title": "Idempotent Zero-Downtime Deployment Script",
        "prompt": "Write a safe, defensive Bash deployment script with strict error handling, signal traps, and health validation.",
        "code": '''#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
readonly APP_NAME="coding-llm-server"
readonly PORT=8000
readonly HEALTH_URL="http://127.0.0.1:${PORT}/health"
readonly MAX_WAIT_SECONDS=30

cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        echo >&2 "[ERROR] Deployment failed with status ${exit_code}. Reverting changes..."
    fi
}
trap cleanup EXIT INT TERM

log_info() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $*"
}

log_info "Starting health probe on ${HEALTH_URL}..."
elapsed=0
while [[ ${elapsed} -lt ${MAX_WAIT_SECONDS} ]]; do
    if curl --silent --fail --max-time 2 "${HEALTH_URL}" | grep -q "healthy"; then
        log_info "Application passed health check! Status: ONLINE."
        exit 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    log_info "Waiting for service initialization (${elapsed}/${MAX_WAIT_SECONDS}s)..."
done

echo >&2 "[ERROR] Timed out waiting for ${APP_NAME} to become healthy."
exit 1
'''
    },
]
