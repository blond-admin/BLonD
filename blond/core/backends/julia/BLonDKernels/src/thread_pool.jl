# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# The thread pool of the chunked CPU loops (see `sweep_chunks!`).
#
# Its workers are dedicated operating-system threads, created through libuv
# and adopted by Julia as *foreign* threads: Julia's task scheduler never
# places tasks on them, and they never occupy one of its threads. They claim
# chunks one at a time from an atomic counter, and the calling thread claims
# chunks as well.
#
# Why not the obvious alternatives, measured on a 12-thread desktop with a
# kick of 3e4 macro-particles (14 µs of work):
#
#   * Julia tasks (`Threads.@spawn`, as the CPU histogram uses for its
#     slices) sleep between calls and cost 70–150 µs to wake, which dwarfs
#     every kernel below ~1e6 particles.
#   * A pool with statically assigned chunks (Polyester's `@batch`) is fast
#     while the machine is idle, but a single call takes 10–16 ms as soon as
#     one other process keeps a core busy: the call waits for whichever
#     worker the operating system descheduled. Claiming chunks dynamically
#     removes that failure mode -- a descheduled worker holds no work.
#
# Idle workers spin for a bounded time and then sleep in a futex (on Linux)
# or on a libuv condition variable (anywhere else). Every spin loop passes
# through a `GC.safepoint()` and every blocking call is `gc_safe`, so a
# sleeping worker never holds up a garbage collection.

"""
    POOL_SUPPORTED

Whether [`run_chunks!`] uses [`ChunkThreadPool`]. Sleeping in the pool
requires `gc_safe` foreign calls, without which a sleeping worker would
block every garbage collection; older Julia versions run the chunks on
tasks instead.
"""
const POOL_SUPPORTED = VERSION >= v"1.12"

"""
    Sleeper

How the threads of a [`ChunkThreadPool`] sleep and wake each other.
[`FutexSleeper`] (Linux) and [`CondvarSleeper`] (any platform) are
interchangeable.
"""
abstract type Sleeper end

"""
    FutexSleeper()

Sleeper using the Linux `futex` system call directly: threads sleep on the
pool's own atomic counters, with no intermediate lock.
"""
struct FutexSleeper <: Sleeper end

const FUTEX_SYSCALL_NUMBER =
    Sys.ARCH === :x86_64 ? 202 : Sys.ARCH === :aarch64 ? 98 : -1
const FUTEX_WAIT_PRIVATE = Cint(128)
const FUTEX_WAKE_PRIVATE = Cint(129)

"""
    futex_supported() -> Bool

Whether this platform offers the `futex` system call at a known number.
"""
futex_supported()::Bool = Sys.islinux() && FUTEX_SYSCALL_NUMBER > 0

"""
    futex_word(atomic) -> Ptr{UInt32}

Address of the 32-bit value inside `atomic`, the word a futex waits on.
"""
futex_word(atomic::Threads.Atomic{UInt32}) =
    Ptr{UInt32}(pointer_from_objref(atomic))

"""
    futex_wait(atomic, expected)

Sleep until `atomic` differs from `expected` and a waker signals it.

The system call returns at once if the word no longer holds `expected`, so
a wake-up between the caller's own check and this call is never lost.
"""
function futex_wait(atomic::Threads.Atomic{UInt32}, expected::UInt32)
    @static if POOL_SUPPORTED
        # `gc_safe` lets a garbage collection proceed while this thread
        # sleeps here.
        @ccall gc_safe = true syscall(
            FUTEX_SYSCALL_NUMBER::Clong;
            futex_word(atomic)::Ptr{UInt32},
            FUTEX_WAIT_PRIVATE::Cint,
            expected::UInt32,
            C_NULL::Ptr{Cvoid},
            C_NULL::Ptr{Cvoid},
            UInt32(0)::UInt32,
        )::Clong
    end
    return nothing
end

"""
    futex_wake(atomic, n_threads)

Wake at most `n_threads` threads sleeping on `atomic`.
"""
function futex_wake(atomic::Threads.Atomic{UInt32}, n_threads::Integer)
    @ccall syscall(
        FUTEX_SYSCALL_NUMBER::Clong;
        futex_word(atomic)::Ptr{UInt32},
        FUTEX_WAKE_PRIVATE::Cint,
        Cint(min(n_threads, typemax(Cint)))::Cint,
    )::Clong
    return nothing
end

"""
    CondvarSleeper()

Portable sleeper: one libuv mutex and one condition variable per side
(workers waiting for work, caller waiting for the workers).
"""
struct CondvarSleeper <: Sleeper
    mutex::Ptr{Cvoid}
    work_condition::Ptr{Cvoid}
    idle_condition::Ptr{Cvoid}
end

function CondvarSleeper()
    # 256 bytes exceed `uv_mutex_t` and `uv_cond_t` on every platform.
    mutex = Libc.malloc(256)
    work_condition = Libc.malloc(256)
    idle_condition = Libc.malloc(256)
    @ccall uv_mutex_init(mutex::Ptr{Cvoid})::Cint
    @ccall uv_cond_init(work_condition::Ptr{Cvoid})::Cint
    @ccall uv_cond_init(idle_condition::Ptr{Cvoid})::Cint
    return CondvarSleeper(mutex, work_condition, idle_condition)
end

function lock_mutex(sleeper::CondvarSleeper)
    @static if POOL_SUPPORTED
        @ccall gc_safe = true uv_mutex_lock(sleeper.mutex::Ptr{Cvoid})::Cvoid
    end
    return nothing
end

unlock_mutex(sleeper::CondvarSleeper) =
    @ccall uv_mutex_unlock(sleeper.mutex::Ptr{Cvoid})::Cvoid

function wait_condition(sleeper::CondvarSleeper, condition::Ptr{Cvoid})
    @static if POOL_SUPPORTED
        @ccall gc_safe = true uv_cond_wait(
            condition::Ptr{Cvoid}, sleeper.mutex::Ptr{Cvoid}
        )::Cvoid
    end
    return nothing
end

signal_condition(condition::Ptr{Cvoid}) =
    @ccall uv_cond_signal(condition::Ptr{Cvoid})::Cvoid
broadcast_condition(condition::Ptr{Cvoid}) =
    @ccall uv_cond_broadcast(condition::Ptr{Cvoid})::Cvoid

"""
    ChunkJob(range_function!, n_elements, chunk_size, n_chunks,
             chunks_per_claim, arguments)

One call of [`run_chunks!`]: the loop body and everything it needs.

`chunks_per_claim` is how many chunks a thread takes out of the counter at
once, see [`chunks_per_claim`](@ref).
"""
struct ChunkJob{RangeFunction, Arguments <: Tuple}
    range_function!::RangeFunction
    n_elements::Int
    chunk_size::Int
    n_chunks::Int
    chunks_per_claim::Int
    arguments::Arguments
end

"""
    CLAIMS_PER_THREAD

How often each thread claims work within one job. Claiming is a contended
atomic on a single counter, about 13 ns a time: claiming every chunk on its
own cost 12.7 µs of a 1e6-particle loop (977 chunks), which is most of that
loop. A handful of claims per thread keeps that overhead at a few hundred
nanoseconds and still lets a thread that falls behind hand its remaining
work to the others.
"""
const CLAIMS_PER_THREAD = 4

"""
    chunks_per_claim(n_chunks, n_threads) -> Int

Chunks a thread takes per claim, so that each of `n_threads` threads makes
about `CLAIMS_PER_THREAD` claims.
"""
chunks_per_claim(n_chunks::Int, n_threads::Int)::Int =
    max(1, cld(n_chunks, CLAIMS_PER_THREAD * n_threads))

"""
    ChunkThreadPool(sleeper; n_workers, worker_spin_seconds,
                    caller_spin_seconds)

Pool of `n_workers` worker threads for [`run_chunks!`].

`generation` is the workers' wake-up word, bumped once per job (and at
shutdown); `busy_workers` is the caller's, counting the workers inside the
current job. Both spin for the given time before they sleep.
"""
mutable struct ChunkThreadPool{S <: Sleeper}
    const sleeper::S
    const n_workers::Int
    const worker_spin_nanoseconds::UInt64
    const caller_spin_nanoseconds::UInt64
    const generation::Threads.Atomic{UInt32}
    const busy_workers::Threads.Atomic{UInt32}
    const sleeping_workers::Threads.Atomic{Int}
    const caller_sleeping::Threads.Atomic{Bool}
    const job_open::Threads.Atomic{Bool}
    const next_chunk::Threads.Atomic{Int}
    const in_use::Threads.Atomic{Bool}
    const shutting_down::Threads.Atomic{Bool}
    @atomic job::Any
    @atomic failure::Any
    const thread_handles::Vector{UInt}
end

function ChunkThreadPool(
    sleeper::Sleeper;
    n_workers::Int,
    worker_spin_seconds::Real,
    caller_spin_seconds::Real,
)
    return ChunkThreadPool(
        sleeper,
        n_workers,
        round(UInt64, worker_spin_seconds * 1e9),
        round(UInt64, caller_spin_seconds * 1e9),
        Threads.Atomic{UInt32}(0),
        Threads.Atomic{UInt32}(0),
        Threads.Atomic{Int}(0),
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Int}(0),
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Bool}(false),
        nothing,
        nothing,
        zeros(UInt, n_workers),
    )
end

# Every running pool, so that a worker thread can find its own pool by
# index; the list also keeps the pools from being collected.
const RUNNING_POOLS = ChunkThreadPool[]
const RUNNING_POOLS_LOCK = ReentrantLock()

@inline cpu_pause() = ccall(:jl_cpu_pause, Cvoid, ())

"""
    sleep_until_new_generation(pool, seen_generation)

Sleep until `pool.generation` leaves `seen_generation`.
"""
function sleep_until_new_generation(
    pool::ChunkThreadPool{FutexSleeper}, seen_generation::UInt32
)
    Threads.atomic_add!(pool.sleeping_workers, 1)
    if pool.generation[] == seen_generation
        futex_wait(pool.generation, seen_generation)
    end
    Threads.atomic_sub!(pool.sleeping_workers, 1)
    return nothing
end

function sleep_until_new_generation(
    pool::ChunkThreadPool{CondvarSleeper}, seen_generation::UInt32
)
    sleeper = pool.sleeper
    lock_mutex(sleeper)
    Threads.atomic_add!(pool.sleeping_workers, 1)
    while pool.generation[] == seen_generation
        wait_condition(sleeper, sleeper.work_condition)
    end
    Threads.atomic_sub!(pool.sleeping_workers, 1)
    unlock_mutex(sleeper)
    return nothing
end

"""
    wake_caller_if_idle(pool)

Leave the current job and wake the caller if this was its last worker.
"""
function wake_caller_if_idle(pool::ChunkThreadPool{FutexSleeper})
    if Threads.atomic_sub!(pool.busy_workers, UInt32(1)) == 1 &&
       pool.caller_sleeping[]
        futex_wake(pool.busy_workers, 1)
    end
    return nothing
end

function wake_caller_if_idle(pool::ChunkThreadPool{CondvarSleeper})
    if Threads.atomic_sub!(pool.busy_workers, UInt32(1)) == 1 &&
       pool.caller_sleeping[]
        lock_mutex(pool.sleeper)
        signal_condition(pool.sleeper.idle_condition)
        unlock_mutex(pool.sleeper)
    end
    return nothing
end

"""
    wait_for_new_generation(pool, seen_generation) -> UInt32

Spin, then sleep, until the pool's generation differs from
`seen_generation`, and return the new one.
"""
function wait_for_new_generation(pool::ChunkThreadPool, seen_generation::UInt32)
    deadline = time_ns() + pool.worker_spin_nanoseconds
    n_pauses = 0
    while true
        generation = pool.generation[]
        generation != seen_generation && return generation
        if pool.worker_spin_nanoseconds == 0
            sleep_until_new_generation(pool, seen_generation)
            continue
        end
        cpu_pause()
        n_pauses += 1
        if n_pauses & 63 == 0
            GC.safepoint()
            if time_ns() > deadline
                sleep_until_new_generation(pool, seen_generation)
                deadline = time_ns() + pool.worker_spin_nanoseconds
            end
        end
    end
end

"""
    run_share!(pool, job)

Claim and run chunks of `job` until none are left.

A chunk that throws stores its exception (the first one wins) and cancels
the chunks nobody has claimed yet; [`run_chunks!`] rethrows it in the
caller once every running chunk has returned.
"""
function run_share!(pool::ChunkThreadPool, job::ChunkJob)
    try
        while true
            claimed = Threads.atomic_add!(
                pool.next_chunk, job.chunks_per_claim
            )
            first_claimed = claimed + 1
            first_claimed > job.n_chunks && break
            last_claimed = min(
                claimed + job.chunks_per_claim, job.n_chunks
            )
            for chunk in first_claimed:last_claimed
                first_element = (chunk - 1) * job.chunk_size + 1
                last_element = min(chunk * job.chunk_size, job.n_elements)
                job.range_function!(
                    first_element, last_element, job.arguments...
                )
                GC.safepoint()
            end
        end
    catch exception
        captured = CapturedException(exception, catch_backtrace())
        @atomicreplace pool.failure nothing => captured
        pool.next_chunk[] = typemax(Int) ÷ 2
    end
    return nothing
end

"""
    worker_loop(pool)

Wait for jobs and run a share of each, until the pool shuts down.
"""
function worker_loop(pool::ChunkThreadPool)
    seen_generation = pool.generation[]
    while true
        seen_generation = wait_for_new_generation(pool, seen_generation)
        pool.shutting_down[] && return nothing
        Threads.atomic_add!(pool.busy_workers, UInt32(1))
        if pool.job_open[]
            # One dynamic dispatch per worker and job; the chunk loop inside
            # `run_share!` is then specialised on the job's types.
            run_share!(pool, @atomic(pool.job))
        end
        wake_caller_if_idle(pool)
    end
end

"""
    worker_entry(pool_index_pointer)

Entry point of a worker's operating-system thread.

Calling this `@cfunction` from a foreign thread adopts that thread into
Julia's foreign thread pool, which is what lets the worker run Julia code
without occupying one of Julia's own threads.
"""
function worker_entry(pool_index_pointer::Ptr{Cvoid})::Cvoid
    pool_index = Int(pool_index_pointer)
    pool = lock(() -> RUNNING_POOLS[pool_index], RUNNING_POOLS_LOCK)
    try
        # One branch per sleeper type, so that the loop below is entered
        # with a concrete pool type rather than through a dispatch.
        if pool isa ChunkThreadPool{FutexSleeper}
            worker_loop(pool)
        elseif pool isa ChunkThreadPool{CondvarSleeper}
            worker_loop(pool)
        end
    catch exception
        # Exceptions of the loop bodies are caught in `run_share!`; report
        # anything else instead of letting the thread die silently.
        ccall(
            :jl_safe_printf, Cvoid, (Cstring,),
            "BLonDKernels: a thread-pool worker died: " *
            "$(sprint(showerror, exception))\n",
        )
    end
    return nothing
end

"""
    start_workers!(pool) -> pool

Create the pool's worker threads.
"""
function start_workers!(pool::ChunkThreadPool)
    pool_index = lock(RUNNING_POOLS_LOCK) do
        push!(RUNNING_POOLS, pool)
        length(RUNNING_POOLS)
    end
    entry = @cfunction(worker_entry, Cvoid, (Ptr{Cvoid},))
    for worker in 1:(pool.n_workers)
        handle = Ref{UInt}(0)
        status = @ccall uv_thread_create(
            handle::Ref{UInt},
            entry::Ptr{Cvoid},
            Ptr{Cvoid}(pool_index)::Ptr{Cvoid},
        )::Cint
        status == 0 || error("uv_thread_create failed with status $status")
        pool.thread_handles[worker] = handle[]
    end
    return pool
end

"""
    stop_workers!(pool)

Wake every worker, let them leave their loops, and join their threads.
"""
function stop_workers!(pool::ChunkThreadPool)
    pool.shutting_down[] = true
    Threads.atomic_add!(pool.generation, UInt32(1))
    wake_all(pool)
    for worker in 1:(pool.n_workers)
        handle = Ref{UInt}(pool.thread_handles[worker])
        @static if POOL_SUPPORTED
            @ccall gc_safe = true uv_thread_join(handle::Ref{UInt})::Cint
        end
    end
    return nothing
end

"""
    wake_workers(pool, n_wanted)

Wake up to `n_wanted` sleeping workers.
"""
function wake_workers(pool::ChunkThreadPool{FutexSleeper}, n_wanted::Int)
    pool.sleeping_workers[] > 0 && futex_wake(pool.generation, n_wanted)
    return nothing
end

function wake_workers(pool::ChunkThreadPool{CondvarSleeper}, n_wanted::Int)
    if pool.sleeping_workers[] > 0
        lock_mutex(pool.sleeper)
        if n_wanted >= pool.n_workers
            broadcast_condition(pool.sleeper.work_condition)
        else
            for _ in 1:n_wanted
                signal_condition(pool.sleeper.work_condition)
            end
        end
        unlock_mutex(pool.sleeper)
    end
    return nothing
end

wake_all(pool::ChunkThreadPool) = wake_workers(pool, Int(typemax(Cint)))

"""
    sleep_until_idle(pool)

Sleep until no worker is inside the current job.
"""
function sleep_until_idle(pool::ChunkThreadPool{FutexSleeper})
    while true
        busy_workers = pool.busy_workers[]
        busy_workers == 0 && return nothing
        futex_wait(pool.busy_workers, busy_workers)
    end
end

function sleep_until_idle(pool::ChunkThreadPool{CondvarSleeper})
    lock_mutex(pool.sleeper)
    while pool.busy_workers[] != 0
        wait_condition(pool.sleeper, pool.sleeper.idle_condition)
    end
    unlock_mutex(pool.sleeper)
    return nothing
end

"""
    wait_until_idle(pool)

Spin, then sleep, until no worker is inside the current job.
"""
function wait_until_idle(pool::ChunkThreadPool)
    deadline = time_ns() + pool.caller_spin_nanoseconds
    n_pauses = 0
    while pool.busy_workers[] != 0
        cpu_pause()
        n_pauses += 1
        if n_pauses & 63 == 0
            GC.safepoint()
            time_ns() > deadline && break
        end
    end
    pool.caller_sleeping[] = true
    sleep_until_idle(pool)
    pool.caller_sleeping[] = false
    return nothing
end

"""
    serial_chunks!(range_function!, n_elements, chunk_size, arguments)

Run every chunk on the calling thread.
"""
function serial_chunks!(
    range_function!, n_elements::Int, chunk_size::Int, arguments::Tuple
)
    for chunk in 1:cld(n_elements, chunk_size)
        first_element = (chunk - 1) * chunk_size + 1
        last_element = min(chunk * chunk_size, n_elements)
        range_function!(first_element, last_element, arguments...)
    end
    return nothing
end

"""
    run_chunks!(pool, range_function!, n_elements, chunk_size, arguments)

Call ``range_function!(first, last, arguments...)`` once per chunk of
`chunk_size` of `n_elements` elements, spread over the pool's workers and
the calling thread.

An exception thrown by a chunk is rethrown here as a `CapturedException`
once every running chunk has returned; the chunks nobody had claimed are
skipped. A nested or concurrent call (the pool is already running a job)
runs its chunks on its own thread, so that the pool cannot deadlock on
itself.
"""
function run_chunks!(
    pool::ChunkThreadPool,
    range_function!,
    n_elements::Int,
    chunk_size::Int,
    arguments::Tuple,
)
    n_chunks = cld(n_elements, chunk_size)
    if n_chunks <= 1 || pool.n_workers == 0 ||
       Threads.atomic_cas!(pool.in_use, false, true)
        serial_chunks!(range_function!, n_elements, chunk_size, arguments)
        return nothing
    end
    job = ChunkJob(
        range_function!,
        n_elements,
        chunk_size,
        n_chunks,
        chunks_per_claim(n_chunks, pool.n_workers + 1),
        arguments,
    )
    @atomic pool.job = job
    pool.next_chunk[] = 0
    pool.job_open[] = true
    # The generation is bumped last: a worker that wakes on it sees the job.
    Threads.atomic_add!(pool.generation, UInt32(1))
    wake_workers(pool, n_chunks - 1)
    run_share!(pool, job)
    pool.job_open[] = false
    wait_until_idle(pool)
    @atomic pool.job = nothing
    failure = @atomic pool.failure
    failure === nothing || (@atomic pool.failure = nothing)
    pool.in_use[] = false
    failure === nothing || throw(failure)
    return nothing
end

"""
    physical_core_count() -> Int

Number of physical CPU cores, or `Sys.CPU_THREADS` where they cannot be
counted. On Linux the kernel lists the hardware threads sharing a core in
`thread_siblings_list`, so counting the distinct lists counts the cores.
"""
function physical_core_count()::Int
    topology = "/sys/devices/system/cpu"
    isdir(topology) || return Sys.CPU_THREADS
    cores = Set{String}()
    for cpu in readdir(topology)
        siblings = joinpath(topology, cpu, "topology/thread_siblings_list")
        isfile(siblings) || continue
        push!(cores, strip(read(siblings, String)))
    end
    return isempty(cores) ? Sys.CPU_THREADS : length(cores)
end

"""
    default_worker_count() -> Int

Workers of the default pool: one per physical core besides the calling
thread, and never more than Julia has threads (so a single-threaded session
stays serial).

One worker per core rather than per hardware thread: the loops are bound by
memory bandwidth often enough that the second thread of a core adds
contention instead of throughput. Measured per element over the four
particle loops at ``1e4``–``1e7`` elements, 11 workers against 5 on a
6-core machine: 0.54–1.15x, i.e. mostly slower, and never much faster.
"""
default_worker_count()::Int = max(
    min(Threads.nthreads(:default), physical_core_count()) - 1, 0
)

const DEFAULT_SLEEPER_TYPE =
    futex_supported() ? FutexSleeper : CondvarSleeper
const DEFAULT_POOL = Ref{ChunkThreadPool{DEFAULT_SLEEPER_TYPE}}()
const DEFAULT_POOL_LOCK = ReentrantLock()

make_sleeper(::Type{FutexSleeper}) = FutexSleeper()
make_sleeper(::Type{CondvarSleeper}) = CondvarSleeper()

"""
    POOL_SPIN_SECONDS

How long a worker or the caller spins before it sleeps. One millisecond
keeps back-to-back calls from sleeping at all, while a pause between calls
costs one wake-up rather than a millisecond of spinning per thread: 5 ms
lowered the latency after idle gaps but tripled the interference with
Julia's own task threads (the CPU histogram), and spinning not at all
produced occasional multi-second outliers.
"""
const POOL_SPIN_SECONDS = 1e-3

"""
    default_pool() -> ChunkThreadPool

The process-wide pool of [`run_chunks!`], started on first use.
"""
function default_pool()
    isassigned(DEFAULT_POOL) && return DEFAULT_POOL[]
    return lock(DEFAULT_POOL_LOCK) do
        if !isassigned(DEFAULT_POOL)
            pool = ChunkThreadPool(
                make_sleeper(DEFAULT_SLEEPER_TYPE);
                n_workers=default_worker_count(),
                worker_spin_seconds=POOL_SPIN_SECONDS,
                caller_spin_seconds=POOL_SPIN_SECONDS,
            )
            start_workers!(pool)
            atexit(() -> stop_workers!(pool))
            DEFAULT_POOL[] = pool
        end
        DEFAULT_POOL[]
    end
end

"""
    run_chunks_on_tasks!(range_function!, n_elements, chunk_size, arguments)

Run the chunks on Julia tasks, one block of chunks per thread.

The fallback for Julia versions without `gc_safe` foreign calls, where the
pool cannot sleep safely.
"""
function run_chunks_on_tasks!(
    range_function!, n_elements::Int, chunk_size::Int, arguments::Tuple
)
    n_chunks = cld(n_elements, chunk_size)
    n_blocks = min(Threads.nthreads(), n_chunks)
    if n_blocks <= 1
        serial_chunks!(range_function!, n_elements, chunk_size, arguments)
        return nothing
    end
    chunks_per_block = cld(n_chunks, n_blocks)
    tasks = map(1:n_blocks) do block
        first_element = (block - 1) * chunks_per_block * chunk_size + 1
        last_element = min(block * chunks_per_block * chunk_size, n_elements)
        Threads.@spawn range_function!(
            first_element, last_element, arguments...
        )
    end
    foreach(wait, tasks)
    return nothing
end

"""
    run_chunks!(range_function!, n_elements, chunk_size, arguments)

Run the chunks on the default pool, or on tasks where the pool is not
supported (see [`POOL_SUPPORTED`]).
"""
function run_chunks!(
    range_function!, n_elements::Int, chunk_size::Int, arguments::Tuple
)
    @static if POOL_SUPPORTED
        run_chunks!(
            default_pool(), range_function!, n_elements, chunk_size, arguments
        )
    else
        run_chunks_on_tasks!(range_function!, n_elements, chunk_size, arguments)
    end
    return nothing
end
