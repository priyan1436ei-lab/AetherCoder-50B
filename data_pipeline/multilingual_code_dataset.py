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

    # -------------------------------------------------------------
    # 10. Multi-File Reasoning & Clean Architecture (Python)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="py-002-clean-arch-repository",
        domain="multi_file_reasoning",
        language="python",
        prompt="Design a clean architecture multi-file structure with Domain Entity, Repository Interface, and Async SQLAlchemy implementation.",
        code="""# File 1: domain/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class UserEntity:
    user_id: str
    email: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = datetime.utcnow()

# File 2: domain/repositories.py
from abc import ABC, abstractmethod
from typing import Optional, List

class UserRepository(ABC):
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[UserEntity]:
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[UserEntity]:
        pass

    @abstractmethod
    async def save(self, user: UserEntity) -> UserEntity:
        pass

# File 3: infrastructure/sql_user_repo.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

class SqlAlchemyUserRepository(UserRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, user_id: str) -> Optional[UserEntity]:
        query = select(UserEntity).where(UserEntity.user_id == user_id)
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[UserEntity]:
        query = select(UserEntity).where(UserEntity.email == email)
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def save(self, user: UserEntity) -> UserEntity:
        self._session.add(user)
        await self._session.flush()
        return user
""",
        tests="""import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_user_repository_interface():
    mock_session = AsyncMock()
    repo = SqlAlchemyUserRepository(mock_session)
    assert isinstance(repo, UserRepository)
""",
    ),

    # -------------------------------------------------------------
    # 11. Production Unit Testing Suite (Python PyTest)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="py-003-unit-test-suite",
        domain="unit_test",
        language="python",
        prompt="Write a comprehensive unit test suite using Pytest with fixtures, parametrization, exception handling, and mock patching.",
        code="""import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

class InsufficientFundsError(Exception):
    pass

class BankAccount:
    def __init__(self, account_id: str, initial_balance: Decimal = Decimal("0.00")):
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.account_id = account_id
        self._balance = initial_balance

    @property
    def balance(self) -> Decimal:
        return self._balance

    def deposit(self, amount: Decimal) -> Decimal:
        if amount <= Decimal("0.00"):
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        return self._balance

    def withdraw(self, amount: Decimal) -> Decimal:
        if amount <= Decimal("0.00"):
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise InsufficientFundsError(f"Cannot withdraw {amount}: balance is {self._balance}")
        self._balance -= amount
        return self._balance

# Test Suite
@pytest.fixture
def account():
    return BankAccount(account_id="acc_001", initial_balance=Decimal("100.00"))

@pytest.mark.parametrize("deposit_val,expected_balance", [
    (Decimal("10.00"), Decimal("110.00")),
    (Decimal("0.50"), Decimal("100.50")),
    (Decimal("1000.00"), Decimal("1100.00")),
])
def test_valid_deposits(account, deposit_val, expected_balance):
    assert account.deposit(deposit_val) == expected_balance
    assert account.balance == expected_balance

def test_invalid_deposit_raises(account):
    with pytest.raises(ValueError, match="Deposit amount must be positive"):
        account.deposit(Decimal("-5.00"))

def test_withdraw_insufficient_funds_raises(account):
    with pytest.raises(InsufficientFundsError):
        account.withdraw(Decimal("200.00"))
""",
        tests="""# All tests self-contained in snippet
def test_run_verification():
    acc = BankAccount("test", Decimal("50.00"))
    acc.deposit(Decimal("25.00"))
    assert acc.balance == Decimal("75.00")
""",
    ),

    # -------------------------------------------------------------
    # 12. Security: SSRF Protection with DNS Pinning (TypeScript)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="sec-002-ssrf-dns-pinning",
        domain="security",
        language="typescript",
        prompt="Prevent Server-Side Request Forgery (SSRF) by validating target IP addresses and blocking private CIDR blocks.",
        code="""// Vulnerable Code:
import http from 'http';

export async function proxyWebhook(targetUrl: string, payload: any) {
  // VULNERABLE: Direct fetch allows attackers to access cloud metadata (169.254.169.254) or localhost
  return fetch(targetUrl, {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  });
}
""",
        target_fix="""// Secure Remediated Code:
import dns from 'dns/promises';
import net from 'net';

const PRIVATE_IPV4_RANGES = [
  /^127\\./,               // Loopback
  /^10\\./,                // Class A private
  /^172\\.(1[6-9]|2[0-9]|3[0-1])\\./, // Class B private
  /^192\\.168\\./,          // Class C private
  /^169\\.254\\./,          // Link-local / Cloud Metadata
];

export async function validateSafeUrl(rawUrl: string): Promise<URL> {
  const parsed = new URL(rawUrl);
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error('Disallowed protocol. Only HTTP(S) permitted.');
  }

  const host = parsed.hostname;
  const lookup = await dns.lookup(host);
  const ip = lookup.address;

  if (lookup.family === 4) {
    for (const pattern of PRIVATE_IPV4_RANGES) {
      if (pattern.test(ip)) {
        throw new Error(`SSRF Blocked: Target resolves to prohibited private IP: ${ip}`);
      }
    }
  } else if (lookup.family === 6 && (ip === '::1' || ip.startsWith('fc00:') || ip.startsWith('fe80:'))) {
    throw new Error(`SSRF Blocked: Prohibited IPv6 private address: ${ip}`);
  }

  return parsed;
}
""",
    ),

    # -------------------------------------------------------------
    # 13. Security: SQL Injection Remediation (Python)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="sec-003-sqli-parameterized",
        domain="security",
        language="python",
        prompt="Remediate SQL injection vulnerability by enforcing parameterized query execution.",
        code="""# Vulnerable Code:
def query_user_orders(db_cursor, username: str, status_filter: str):
    # VULNERABLE: String formatting permits SQL Injection: ' OR 1=1 --
    query = f"SELECT id, total, status FROM orders WHERE username = '{username}' AND status = '{status_filter}'"
    db_cursor.execute(query)
    return db_cursor.fetchall()
""",
        target_fix="""# Secure Code:
def query_user_orders(db_cursor, username: str, status_filter: str):
    \"\"\"Remediated: Uses parameterized query syntax preventing SQL injection.\"\"\"
    query = "SELECT id, total, status FROM orders WHERE username = %s AND status = %s"
    db_cursor.execute(query, (username, status_filter))
    return db_cursor.fetchall()
""",
    ),

    # -------------------------------------------------------------
    # 14. Debugging: Goroutine Channel Deadlock (Go)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="debug-002-go-channel-deadlock",
        domain="debugging",
        language="go",
        prompt="Fix the deadlock in this asynchronous pipeline caused by unbuffered channel write without a reader.",
        code="""// Buggy Code:
package pipeline

func ProcessData(values []int) []int {
	ch := make(chan int) // Unbuffered: Blocks sender when no reader is active
	var out []int

	for _, v := range values {
		ch <- v * 2 // Deadlocks on second iteration!
	}
	close(ch)

	for res := range ch {
		out = append(out, res)
	}
	return out
}
""",
        target_fix="""// Fixed Code:
package pipeline

func ProcessData(values []int) []int {
	out := make([]int, 0, len(values))
	if len(values) == 0 {
		return out
	}

	ch := make(chan int, len(values)) // Buffered to hold all outputs without blocking
	for _, v := range values {
		ch <- v * 2
	}
	close(ch)

	for res := range ch {
		out = append(out, res)
	}
	return out
}
""",
    ),

    # -------------------------------------------------------------
    # 15. Instruction Following: Token Bucket Rate Limiter (Rust)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="rs-002-token-bucket-ratelimiter",
        domain="instruction_following",
        language="rust",
        prompt="Implement a high-performance thread-safe Token Bucket Rate Limiter in Rust with atomics.",
        code="""use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct TokenBucket {
    capacity: u64,
    refill_rate_per_sec: u64,
    tokens: AtomicU64,
    last_refill_instant: Instant,
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate_per_sec: u64) -> Self {
        Self {
            capacity,
            refill_rate_per_sec,
            tokens: AtomicU64::new(capacity),
            last_refill_instant: Instant::now(),
        }
    }

    pub fn try_acquire(&self, count: u64) -> bool {
        let mut current = self.tokens.load(Ordering::Acquire);
        loop {
            if current < count {
                return false;
            }
            match self.tokens.compare_exchange_weak(
                current,
                current - count,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(updated) => current = updated,
            }
        }
    }
}
""",
        tests="""#[test]
fn test_rate_limiter() {
    let bucket = TokenBucket::new(10, 10);
    assert!(bucket.try_acquire(5));
    assert!(bucket.try_acquire(5));
    assert!(!bucket.try_acquire(1));
}
""",
    ),

    # -------------------------------------------------------------
    # 16. General: Consistent Hash Ring (Go)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="go-002-consistent-hash-ring",
        domain="general",
        language="go",
        prompt="Implement a consistent hash ring with virtual nodes in Go for distributed load balancing.",
        code="""package hashing

import (
	"fmt"
	"hash/fnv"
	"sort"
	"strconv"
	"sync"
)

type HashRing struct {
	mu           sync.RWMutex
	replicas     int
	ring         []uint32
	ringMap      map[uint32]string
}

func NewHashRing(replicas int) *HashRing {
	return &HashRing{
		replicas: replicas,
		ringMap:  make(map[uint32]string),
	}
}

func (h *HashRing) hashKey(key string) uint32 {
	hasher := fnv.New32a()
	hasher.Write([]byte(key))
	return hasher.Sum32()
}

func (h *HashRing) AddNode(node string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for i := 0; i < h.replicas; i++ {
		vnodeKey := node + "#" + strconv.Itoa(i)
		hash := h.hashKey(vnodeKey)
		h.ring = append(h.ring, hash)
		h.ringMap[hash] = node
	}
	sort.Slice(h.ring, func(i, j int) bool { return h.ring[i] < h.ring[j] })
}

func (h *HashRing) GetNode(key string) (string, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(h.ring) == 0 {
		return "", fmt.Errorf("hash ring is empty")
	}

	hash := h.hashKey(key)
	idx := sort.Search(len(h.ring), func(i int) bool { return h.ring[i] >= hash })
	if idx == len(h.ring) {
		idx = 0 // Wrap around the ring
	}
	return h.ringMap[h.ring[idx]], nil
}
""",
        tests="""package hashing
import "testing"

func TestHashRing(t *testing.T) {
	ring := NewHashRing(3)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	node, err := ring.GetNode("user_session_482")
	if err != nil || node == "" {
		t.Fatalf("Expected valid node, got: %s, %v", node, err)
	}
}
""",
    ),

    # -------------------------------------------------------------
    # 17. Code Completion: Discriminated State Machine (TypeScript)
    # -------------------------------------------------------------
    CodeSample(
        sample_id="ts-002-state-machine",
        domain="code_completion",
        language="typescript",
        prompt="Implement a type-safe generic state machine with discriminated transitions and entry/exit actions.",
        code="""export type MachineState = 'IDLE' | 'FETCHING' | 'SUCCESS' | 'ERROR';
export type MachineEvent = 
  | { type: 'START' }
  | { type: 'RESOLVE'; data: string[] }
  | { type: 'REJECT'; error: Error }
  | { type: 'RESET' };

export interface MachineContext {
  state: MachineState;
  data?: string[];
  error?: Error;
}

export function transition(ctx: MachineContext, event: MachineEvent): MachineContext {
  switch (ctx.state) {
    case 'IDLE':
      if (event.type === 'START') {
        return { state: 'FETCHING' };
      }
      return ctx;
    case 'FETCHING':
      if (event.type === 'RESOLVE') {
        return { state: 'SUCCESS', data: event.data };
      }
      if (event.type === 'REJECT') {
        return { state: 'ERROR', error: event.error };
      }
      return ctx;
    case 'SUCCESS':
    case 'ERROR':
      if (event.type === 'RESET') {
        return { state: 'IDLE' };
      }
      return ctx;
  }
}
""",
    ),

    # -------------------------------------------------------------
    # 18. Agent Trajectory: Multi-File Cross-Module Refactor
    # -------------------------------------------------------------
    CodeSample(
        sample_id="agent-002-multifile-refactor-import",
        domain="agent_trajectory",
        language="json",
        prompt="Autonomous agent trajectory performing multi-file refactoring and updating dependent import statements.",
        code="""{
  "trajectory_id": "traj-refactor-089",
  "goal": "Refactor UserCredentials into auth/credentials.py and update references in server.py and routes.py",
  "steps": [
    {
      "step": 1,
      "thought": "Examine project files to locate all imports of UserCredentials.",
      "tool_call": {
        "name": "search_code",
        "arguments": {"query": "from models import UserCredentials"}
      },
      "tool_result": ["src/api/routes.py", "src/server.py"]
    },
    {
      "step": 2,
      "thought": "Move class UserCredentials into src/auth/credentials.py and write new module.",
      "tool_call": {
        "name": "write_file",
        "arguments": {
          "rel_path": "src/auth/credentials.py",
          "content": "from pydantic import BaseModel\\n\\nclass UserCredentials(BaseModel):\\n    username: str\\n    password: str\\n"
        }
      },
      "tool_result": {"status": "success", "bytes_written": 105}
    },
    {
      "step": 3,
      "thought": "Update import in src/api/routes.py to use new module location.",
      "tool_call": {
        "name": "apply_replacement",
        "arguments": {
          "rel_path": "src/api/routes.py",
          "target": "from models import UserCredentials",
          "replacement": "from src.auth.credentials import UserCredentials"
        }
      },
      "tool_result": {"success": true, "diff": "--- a/src/api/routes.py\\n+++ b/src/api/routes.py\\n@@ -2,1 +2,1 @@\\n-from models import UserCredentials\\n+from src.auth.credentials import UserCredentials\\n"}
    },
    {
      "step": 4,
      "thought": "Execute pytest to ensure all endpoints pass without import errors.",
      "tool_call": {
        "name": "run_command",
        "arguments": {"cmd_args": ["pytest", "tests/test_api.py"]}
      },
      "tool_result": {"exit_code": 0, "stdout": "8 passed in 0.22s"}
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
