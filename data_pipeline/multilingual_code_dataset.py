"""Curated multilingual code datasets spanning all 8 core software engineering domains."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os

@dataclass
class CodeSample:
    sample_id: str
    domain: str  # e.g., "general", "instruction_following", "debugging", "unit_test", "agent_trajectory", "security"
    language: str  # e.g., "python", "typescript", "rust", "go", "cpp", "java", "sql", "bash"
    prompt: Optional[str]
    code: str
    target_fix: Optional[str] = None
    tests: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


# Comprehensive multilingual samples covering the 8 domains required for 50B coding specialization
CURATED_DATASET: List[CodeSample] = [
    # -------------------------------------------------------------
    # 1. Python: Advanced Concurrency & Data Structures
    # -------------------------------------------------------------
    CodeSample(
        sample_id="py-001-async-pool",
        domain="instruction_following",
        language="python",
        prompt="Implement an asynchronous connection pool with health checks and graceful timeout handling in Python.",
        code="""import asyncio
import time
from typing import List, Optional

class Connection:
    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.last_active = time.monotonic()
        self.is_healthy = True

    async def ping(self) -> bool:
        await asyncio.sleep(0.005)
        return self.is_healthy

class AsyncConnectionPool:
    def __init__(self, max_size: int = 10, timeout: float = 5.0):
        self.max_size = max_size
        self.timeout = timeout
        self._pool: asyncio.Queue[Connection] = asyncio.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> Connection:
        if self._pool.empty() and self._created_count < self.max_size:
            async with self._lock:
                if self._created_count < self.max_size:
                    self._created_count += 1
                    return Connection(self._created_count)

        try:
            conn = await asyncio.wait_for(self._pool.get(), timeout=self.timeout)
            if not await conn.ping():
                return Connection(conn.conn_id)
            return conn
        except asyncio.TimeoutError:
            raise TimeoutError("Exceeded timeout waiting for an available connection.")

    async def release(self, conn: Connection) -> None:
        conn.last_active = time.monotonic()
        await self._pool.put(conn)
""",
        tests="""import pytest
import asyncio

@pytest.mark.asyncio
async def test_pool_acquire_and_release():
    pool = AsyncConnectionPool(max_size=2, timeout=1.0)
    c1 = await pool.acquire()
    c2 = await pool.acquire()
    assert c1.conn_id == 1
    assert c2.conn_id == 2
    await pool.release(c1)
    c3 = await pool.acquire()
    assert c3.conn_id == 1
""",
    ),

    # -------------------------------------------------------------
    # 2. Rust: Lock-Free Ring Buffer & Memory Safety
    # -------------------------------------------------------------
    CodeSample(
        sample_id="rs-001-lockfree-ring",
        domain="general",
        language="rust",
        prompt="Write a lock-free Single-Producer Single-Consumer (SPSC) ring buffer in Rust with atomic memory ordering.",
        code="""use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub struct SpscRingBuffer<T, const N: usize> {
    buffer: [UnsafeCell<Option<T>>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Send, const N: usize> Sync for SpscRingBuffer<T, N> {}

impl<T, const N: usize> SpscRingBuffer<T, N> {
    pub fn new() -> Self {
        assert!(N > 0 && (N & (N - 1)) == 0, "Buffer capacity must be a power of two");
        Self {
            buffer: std::array::from_fn(|_| UnsafeCell::new(None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= N {
            return Err(item); // Buffer full
        }

        let idx = head & (N - 1);
        unsafe { *self.buffer[idx].get() = Some(item); }
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None; // Buffer empty
        }

        let idx = tail & (N - 1);
        let item = unsafe { (*self.buffer[idx].get()).take() };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        item
    }
}
""",
    ),

    # -------------------------------------------------------------
    # 3. TypeScript / JavaScript: Reactive State Store & Deep Equal
    # -------------------------------------------------------------
    CodeSample(
        sample_id="ts-001-reactive-store",
        domain="instruction_following",
        language="typescript",
        prompt="Implement an immutable, type-safe reactive state store with selectors and unsubscribe logic in TypeScript.",
        code="""export type Listener<T> = (state: T, prevState: T) => void;
export type Selector<T, R> = (state: T) => R;

export class ObservableStore<T extends Record<string, any>> {
  private state: T;
  private listeners: Set<Listener<T>> = new Set();

  constructor(initialState: T) {
    this.state = Object.freeze({ ...initialState });
  }

  public getState(): T {
    return this.state;
  }

  public setState(updater: Partial<T> | ((prev: T) => Partial<T>)): void {
    const prevState = this.state;
    const partial = typeof updater === 'function' ? updater(prevState) : updater;
    const nextState = Object.freeze({ ...prevState, ...partial });

    if (!this.shallowEqual(prevState, nextState)) {
      this.state = nextState;
      this.listeners.forEach(listener => listener(nextState, prevState));
    }
  }

  public subscribe(listener: Listener<T>): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public select<R>(selector: Selector<T, R>, onChange: (value: R) => void): () => void {
    let currentVal = selector(this.state);
    return this.subscribe((nextState) => {
      const nextVal = selector(nextState);
      if (nextVal !== currentVal) {
        currentVal = nextVal;
        onChange(nextVal);
      }
    });
  }

  private shallowEqual(objA: any, objB: any): boolean {
    if (Object.is(objA, objB)) return true;
    const keysA = Object.keys(objA);
    const keysB = Object.keys(objB);
    if (keysA.length !== keysB.length) return false;
    return keysA.every(key => Object.is(objA[key], objB[key]));
  }
}
""",
    ),

    # -------------------------------------------------------------
    # 4. Go: Resilient Worker Pool with Graceful Context Cancellation
    # -------------------------------------------------------------
    CodeSample(
        sample_id="go-001-worker-pool",
        domain="instruction_following",
        language="go",
        prompt="Create a production worker pool in Go with worker health metrics and context cancellation.",
        code="""package workerpool

import (
	"context"
	"sync"
	"sync/atomic"
)

type Task[T any, R any] func(ctx context.Context, input T) (R, error)

type Result[R any] struct {
	Value R
	Err   error
}

type Pool[T any, R any] struct {
	numWorkers  int
	tasks       chan T
	results     chan Result[R]
	taskFn      Task[T, R]
	activeTasks int64
	wg          sync.WaitGroup
}

func NewPool[T any, R any](workers int, bufferSize int, fn Task[T, R]) *Pool[T, R] {
	return &Pool[T, R]{
		numWorkers: workers,
		tasks:      make(chan T, bufferSize),
		results:    make(chan Result[R], bufferSize),
		taskFn:     fn,
	}
}

func (p *Pool[T, R]) Start(ctx context.Context) {
	for i := 0; i < p.numWorkers; i++ {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case task, ok := <-p.tasks:
					if !ok {
						return
					}
					atomic.AddInt64(&p.activeTasks, 1)
					val, err := p.taskFn(ctx, task)
					atomic.AddInt64(&p.activeTasks, -1)
					p.results <- Result[R]{Value: val, Err: err}
				}
			}
		}()
	}
}

func (p *Pool[T, R]) Submit(task T) {
	p.tasks <- task
}

func (p *Pool[T, R]) Close() {
	close(p.tasks)
	p.wg.Wait()
	close(p.results)
}
""",
    ),

    # -------------------------------------------------------------
    # 5. C++: High Performance SIMD Vector Math
    # -------------------------------------------------------------
    CodeSample(
        sample_id="cpp-001-simd-dotproduct",
        domain="general",
        language="cpp",
        prompt="Write a vectorized dot product computation in C++ using AVX2 intrinsics with an scalar fallback.",
        code="""#include <cstddef>
#include <immintrin.h>

float vector_dot_product_avx2(const float* a, const float* b, size_t length) {
    size_t i = 0;
    __m256 sum256 = _mm256_setzero_ps();

    // Process 8 floating point numbers simultaneously
    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }

    // Horizontal add of the 8 SIMD lanes
    alignas(32) float buffer[8];
    _mm256_store_ps(buffer, sum256);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // Scalar fallback for remaining elements
    for (; i < length; ++i) {
        total += a[i] * b[i];
    }

    return total;
}
""",
    ),

    # -------------------------------------------------------------
    # 6. SQL: Recursive CTE for Graph Traversal & Partitioning
    # -------------------------------------------------------------
    CodeSample(
        sample_id="sql-001-recursive-org-chart",
        domain="instruction_following",
        language="sql",
        prompt="Write an optimized PostgreSQL query using recursive CTEs to calculate total department payroll and employee hierarchy.",
        code="""WITH RECURSIVE employee_hierarchy AS (
    -- Anchor member: Top-level executives (manager_id is NULL)
    SELECT
        e.employee_id,
        e.full_name,
        e.manager_id,
        e.salary,
        e.department_id,
        1 AS org_depth,
        ARRAY[e.employee_id] AS path_trail
    FROM employees e
    WHERE e.manager_id IS NULL

    UNION ALL

    -- Recursive member: Subordinates
    SELECT
        sub.employee_id,
        sub.full_name,
        sub.manager_id,
        sub.salary,
        sub.department_id,
        eh.org_depth + 1 AS org_depth,
        eh.path_trail || sub.employee_id
    FROM employees sub
    INNER JOIN employee_hierarchy eh ON sub.manager_id = eh.employee_id
    WHERE NOT (sub.employee_id = ANY(eh.path_trail)) -- Prevent circular graph loops
)
SELECT
    eh.department_id,
    COUNT(eh.employee_id) AS team_headcount,
    ROUND(AVG(eh.salary), 2) AS average_compensation,
    SUM(eh.salary) AS total_payroll,
    MAX(eh.org_depth) AS max_hierarchy_depth
FROM employee_hierarchy eh
GROUP BY eh.department_id
ORDER BY total_payroll DESC;
""",
    ),

    # -------------------------------------------------------------
    # 7. Debugging & Root Cause Analysis
    # -------------------------------------------------------------
    CodeSample(
        sample_id="debug-001-python-mutable-default",
        domain="debugging",
        language="python",
        prompt="Identify and fix the bug in this token accumulator function.",
        code="""# Buggy Code:
def accumulate_tokens(token_str: str, registry: list = []) -> list:
    registry.append(token_str)
    return registry
""",
        target_fix="""# Corrected Code:
from typing import List, Optional

def accumulate_tokens(token_str: str, registry: Optional[List[str]] = None) -> List[str]:
    \"\"\"Fixed: Replaced mutable default argument with None to avoid shared state across calls.\"\"\"
    if registry is None:
        registry = []
    registry.append(token_str)
    return registry
""",
    ),

    # -------------------------------------------------------------
    # 8. Secure Coding & Vulnerability Remediation
    # -------------------------------------------------------------
    CodeSample(
        sample_id="sec-001-timing-attack-hmac",
        domain="security",
        language="python",
        prompt="Remediate a timing side-channel attack in API signature verification.",
        code="""# Vulnerable Code:
def verify_signature(received_sig: str, expected_sig: str) -> bool:
    # VULNERABLE: Short-circuiting string equality leaks timing information
    return received_sig == expected_sig
""",
        target_fix="""# Secure Code:
import hmac

def verify_signature(received_sig: str, expected_sig: str) -> bool:
    \"\"\"Remediated: Constant-time comparison prevents timing side-channel attacks.\"\"\"
    if not isinstance(received_sig, str) or not isinstance(expected_sig, str):
        return False
    return hmac.compare_digest(received_sig.encode('utf-8'), expected_sig.encode('utf-8'))
""",
    ),

    # -------------------------------------------------------------
    # 9. Coding Agent Trajectory (ReAct Loop)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="agent-001-react-patch-test",
        domain="agent_trajectory",
        language="json",
        prompt="Trajectory demonstrating autonomous bug localization, patch generation, and test verification.",
        code="""{
  "trajectory_id": "traj-patch-042",
  "goal": "Fix off-by-one index error in binary search",
  "steps": [
    {
      "step": 1,
      "thought": "Inspect repository map to find binary search implementation.",
      "tool_call": {
        "name": "list_files",
        "arguments": {"rel_dir": "algorithms"}
      },
      "tool_result": ["algorithms/search.py", "algorithms/sort.py"]
    },
    {
      "step": 2,
      "thought": "Read algorithms/search.py to locate the boundary check error.",
      "tool_call": {
        "name": "read_file",
        "arguments": {"rel_path": "algorithms/search.py", "start_line": 1, "end_line": 20}
      },
      "tool_result": "def binary_search(arr, target):\\n    low = 0\\n    high = len(arr)\\n    while low <= high:\\n        mid = (low + high) // 2\\n"
    },
    {
      "step": 3,
      "thought": "High index is initialized to len(arr) instead of len(arr) - 1. Apply surgical patch.",
      "tool_call": {
        "name": "apply_replacement",
        "arguments": {
          "rel_path": "algorithms/search.py",
          "target": "high = len(arr)",
          "replacement": "high = len(arr) - 1"
        }
      },
      "tool_result": {"success": true, "diff": "--- a/algorithms/search.py\\n+++ b/algorithms/search.py\\n@@ -3,1 +3,1 @@\\n-    high = len(arr)\\n+    high = len(arr) - 1\\n"}
    },
    {
      "step": 4,
      "thought": "Run pytest suite in sandbox to verify resolution.",
      "tool_call": {
        "name": "run_command",
        "arguments": {"cmd_args": ["pytest", "tests/test_search.py"]}
      },
      "tool_result": {"exit_code": 0, "stdout": "1 passed in 0.04s"}
    }
  ]
}
""",
    ),
]


def save_curated_dataset_jsonl(filepath: str) -> int:
    """Exports the curated dataset samples to a structured JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in CURATED_DATASET:
            sample_dict = {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "language": sample.language,
                "prompt": sample.prompt,
                "code": sample.code,
                "target_fix": sample.target_fix,
                "tests": sample.tests,
            }
            f.write(json.dumps(sample_dict) + "\n")
            count += 1
    return count
