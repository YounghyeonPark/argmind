// SPDX-License-Identifier: Apache-2.0
//
// Bounded sensor streams, ported from argmind/stream.py.
//
// A `Stream` is a fixed-capacity ring of timestamped samples. The capacity is a template
// parameter, so the memory is declared at compile time and nothing here allocates: on an
// embedded target a buffer whose size depends on uptime is a bug rather than a tuning
// parameter.
//
// Overwriting rather than blocking is deliberate, and carried straight across. A sensor
// does not stop producing because the consumer is late, and a framework that blocks the
// producer converts a latency problem into a correctness problem.
//
// The one structural decision, and why
// ------------------------------------
// An `Aligner` has to work across streams of *different* dimensions at once. The obvious
// C++ answer is a base class with a virtual `sample_at`, and this core has no virtuals and
// no heap. The alternative that keeps compile-time dimensions everywhere is a variadic
// `Aligner<Streams...>`, which puts the whole stream set in the type and makes it
// unnameable from a C ABI.
//
// So: `Stream<T, Dim, Capacity>` keeps its dimensions in the type, and hands out a
// `StreamView` — a non-owning descriptor with a *runtime* `dim` — for anything that has to
// treat several streams uniformly. The compile-time dimensions are kept where they earn
// their keep, which is the arithmetic in `rls.hpp`; alignment is a timestamp search and a
// handful of copies at the slow stream's rate, so a runtime `dim` there costs nothing that
// was ever going to matter. That is the same trade the C ABI makes, made once more and for
// the same reason: keep the compile-time shape where it buys inlining, spend it where it
// buys expressiveness.

#ifndef ARGMIND_STREAM_HPP
#define ARGMIND_STREAM_HPP

#include <stddef.h>
#include <stdint.h>

#include "argmind/mat.hpp"

namespace argmind {

/// One observation. `t` is seconds on that stream's own timebase — streams are not assumed
/// to share a clock, and reconciling them is the aligner's job.
///
/// `valid` carries the sensor's own opinion of the reading. Invalid samples are *kept*,
/// because "the sensor said it was bad" and "no sample arrived" are different situations
/// and a consumer usually needs to tell them apart.
template <typename T>
struct SampleRef {
    double t = 0.0;
    const T* values = nullptr;  ///< `dim` of them, pointing into the ring
    bool valid = false;
    bool present = false;       ///< false when a query found nothing
};

/// A non-owning description of a ring, with a runtime dimension.
///
/// Everything the aligner needs and nothing it does not. It is a view: it borrows the
/// storage and is invalidated by nothing except the stream outliving it, which on a target
/// where both are statically allocated is not a scenario.
template <typename T>
struct StreamView {
    const double* times = nullptr;
    const T* values = nullptr;
    const uint8_t* valid = nullptr;
    int dim = 0;
    int capacity = 0;
    int head = 0;    ///< next write position
    int count = 0;   ///< samples currently held
    /// Samples overwritten because the ring was full — carried on the view because
    /// `dropped + count` is the total ever pushed, and that identity is the only honest
    /// basis for a pipeline's lost-frame count. Without it here the pipeline would need the
    /// concrete `Stream<T, Dim, Capacity>`, which is the type it exists to avoid naming.
    uint64_t dropped = 0;

    /// Map age index (0 = oldest held) to a physical slot.
    ///
    /// One conditional add rather than two modulos. Given `0 <= age < count <= capacity`
    /// and `0 <= head < capacity`, the raw index lies in `[head - capacity, capacity)`, so
    /// it is at most one wrap below zero and a single add corrects it. The modulo form ran
    /// two integer divisions per element on every scan; what those cost on the target is
    /// not measured here and no cross-compiler exists on this machine, so the argument is
    /// structural — the integer analogue of the reciprocal rule `mat.hpp` already follows:
    /// replace a division whose operand range is known.
    int slot(int age) const {
        int i = head - count + age;
        if (i < 0) i += capacity;
        return i;
    }

    SampleRef<T> at(int age) const {
        SampleRef<T> s;
        if (age < 0 || age >= count) return s;
        const int i = slot(age);
        s.t = times[i];
        s.values = values + static_cast<size_t>(i) * dim;
        s.valid = valid[i] != 0;
        s.present = true;
        return s;
    }

    /// The most recent sample not later than `t`, or absent.
    ///
    /// The primitive behind hold-last alignment. A linear scan from the newest end:
    /// correct regardless of wrap, and cheap because alignment queries are near the head
    /// in any real-time use.
    SampleRef<T> newest_at_or_before(double t) const {
        for (int age = count - 1; age >= 0; --age) {
            if (times[slot(age)] <= t) return at(age);
        }
        return SampleRef<T>();
    }

    /// The sample closest in time to `t`, before or after.
    SampleRef<T> nearest(double t) const {
        if (count == 0) return SampleRef<T>();
        int best = 0;
        double best_d = -1.0;
        for (int age = 0; age < count; ++age) {
            double d = times[slot(age)] - t;
            if (d < 0) d = -d;
            if (best_d < 0.0 || d < best_d) {
                best = age;
                best_d = d;
            } else if (d > best_d) {
                // Timestamps are strictly increasing, so |t_i - t| falls then rises and no
                // later sample can be closer.
                //
                // Tie resolution is NOT this comparison's doing — an earlier version of
                // this comment claimed it was, and a mutation check disproved it: making
                // the break non-strict changes nothing anyone can observe. What resolves a
                // tie to the *earlier* sample is the strict `<` in the update above, which
                // refuses to move on an equal distance, matching the reference's
                // `d < best_d`. Mutating *that* to `<=` does fail a test.
                break;
            }
        }
        return at(best);
    }

    /// The pair straddling `t`, for interpolation. `after.present` is false when `t` falls
    /// outside the held range — deliberately, rather than extrapolating. Extrapolation past
    /// the end of a sensor stream is a guess, and a fusion framework that guesses without
    /// saying so is how a quiet wrong answer gets shipped.
    bool bracketing(double t, SampleRef<T>* before, SampleRef<T>* after) const {
        SampleRef<T> prev;
        for (int age = 0; age < count; ++age) {
            const SampleRef<T> cur = at(age);
            if (cur.t == t) {
                *before = cur;
                *after = cur;
                return true;
            }
            if (cur.t > t) {
                if (!prev.present) return false;
                *before = prev;
                *after = cur;
                return true;
            }
            prev = cur;
        }
        return false;
    }
};


/// The bookkeeping half of a ring, separated from its storage.
///
/// This exists so that `push` has exactly **one** implementation. `Stream<T, Dim, Capacity>`
/// keeps its shape in the type and the C ABI cannot -- a C caller owns its own buffers and
/// their sizes are runtime values -- so without this there would be two copies of the
/// monotonicity and finiteness refusals. `persist.hpp` already makes the argument that a
/// refusal implemented in two places is one that will eventually disagree with itself, and it
/// applies with more force here: the C side would be the copy, and the copy is the one nobody
/// runs the parity harness against.
///
/// `dim` and `capacity` are parameters rather than template arguments, and that costs nothing
/// where they are constants: `Stream::push` passes `Dim` and `Capacity`, this is `inline`, and
/// the compiler folds them. Measured on x86-64 at `-Os`, `Stream<float,6,64>::push` is the
/// same size before and after the extraction.
template <typename T>
struct Ring {
    T* values = nullptr;         ///< `capacity * dim` elements
    double* times = nullptr;     ///< `capacity`
    uint8_t* valid = nullptr;    ///< `capacity`
    int dim = 0;
    int capacity = 0;
    int head = 0;
    int count = 0;
    uint64_t dropped = 0;
    double last_t = 0.0;
};

/// Append a sample, overwriting the oldest if full. False and no change on a timestamp that
/// is not strictly increasing, or that is +inf.
///
/// The bookkeeping arrives as separate pointers rather than as a `Ring<T>*`, and that is a
/// measured choice rather than a stylistic one. Taking the struct meant `Stream::push` had to
/// materialise one and write four fields back, which cost **32 bytes of code** per
/// instantiation at `-Os` on x86-64 -- 160 to 192 for `Stream<float,6,64>::push`. Passing the
/// members directly brings it back to exactly the pre-refactor 160. The first version of this
/// comment claimed the extraction was free before anything had been measured; it was not, and
/// it took two attempts to make it so.
///
/// Note which check does what, because it differs from the Python's. `!(t > *last_t)` already
/// refuses NaN (no comparison with NaN is true) and -inf (it cannot exceed the initial -inf).
/// The finiteness test therefore exists for exactly one value: **+inf**, which passes the
/// ordering test and then makes every subsequent push impossible, killing the stream silently.
///
/// The Python needed a separate check for a different reason: it wrote `t <= self._last_t`,
/// which is False for NaN, so a NaN was *accepted* and poisoned `_last_t`. Same fix, different
/// hole -- worth saying rather than implying the two sides had the same defect.
template <typename T>
inline bool ring_push(T* __restrict values_, double* __restrict times_,
                      uint8_t* __restrict valid_, int dim, int capacity,
                      int* __restrict head, int* __restrict count,
                      uint64_t* __restrict dropped, double* __restrict last_t,
                      double t, const T* __restrict values, bool valid) {
    if (!(t > *last_t)) return false;
    if (!is_finite(t)) return false;

    if (*count == capacity) ++*dropped;
    const int base = *head * dim;
    for (int j = 0; j < dim; ++j) values_[base + j] = values[j];
    times_[*head] = t;
    valid_[*head] = valid ? 1 : 0;
    // A modulo, where `slot()` twelve lines up deliberately avoids one. The two are not in
    // conflict: this runs ONCE PER PUSH and that ran twice per element of every scan, so the
    // rule there was about an O(count) loop and does not reach here. Measured, and the
    // measurement is why it is a modulo: the conditional-add form costs 16 bytes more at -Os
    // for `Stream<float,6,64>::push`, because a power-of-two `Capacity` folds the modulo to a
    // single AND while the conditional stays a compare and a branch. The C ABI's ring, whose
    // capacity is a runtime value, pays one integer division per push instead -- on the
    // declared floor that is a single SDIV at the sensor's rate, which is not a cost worth
    // making the compile-time path larger to avoid.
    *head = (*head + 1) % capacity;
    if (*count < capacity) ++*count;
    *last_t = t;
    return true;
}

/// The same append against a runtime-sized ring descriptor. One line, so that the C ABI and
/// `Stream<>` cannot drift apart on what a push refuses.
template <typename T>
inline bool ring_push(Ring<T>* __restrict r, double t, const T* __restrict values,
                      bool valid) {
    return ring_push<T>(r->values, r->times, r->valid, r->dim, r->capacity, &r->head,
                        &r->count, &r->dropped, &r->last_t, t, values, valid);
}

template <typename T>
inline StreamView<T> ring_view(const Ring<T>& r) {
    StreamView<T> s;
    s.times = r.times;
    s.values = r.values;
    s.valid = r.valid;
    s.dim = r.dim;
    s.capacity = r.capacity;
    s.head = r.head;
    s.count = r.count;
    s.dropped = r.dropped;
    return s;
}

/// A fixed-capacity ring of samples from one source.
template <typename T, int Dim, int Capacity>
class Stream {
    static_assert(Dim >= 1, "dim must be >= 1");
    static_assert(Capacity >= 1, "capacity must be >= 1");

public:
    using scalar = T;
    static constexpr int dim = Dim;
    static constexpr int capacity = Capacity;

    /// Bytes the ring's *payload* occupies: values, a double of time per slot, and a
    /// validity byte. **Not** the size of the object — the indices, the counter and the
    /// last timestamp add more, and alignment padding adds more again.
    ///
    /// It was called `bytes` and described as what a caller should budget, which was 24
    /// bytes short per stream on this host. A caller budgeting memory wants `sizeof`; this
    /// is here to show where the payload goes. The static assertion below keeps the two
    /// from drifting into disagreement about which is larger.
    static constexpr int payload_bytes =
        Capacity * Dim * static_cast<int>(sizeof(T)) + Capacity * 8 + Capacity;

    Stream() { reset(); }

    void reset() {
        head_ = 0;
        count_ = 0;
        dropped_ = 0;
        last_t_ = -__builtin_inf();
    }

    /// Append a sample, overwriting the oldest if full. Returns false and changes nothing
    /// on a timestamp that is not strictly increasing, or that is +inf.
    ///
    /// The Python raises here. This core has no exceptions, so it reports instead — but the
    /// refusal is the same and for the same reason: a sample arriving before one already
    /// stored means either a clock that went backwards or samples delivered out of order,
    /// and both break every alignment policy downstream. Failing at the source localises
    /// the bug instead of producing a subtly wrong fused estimate somewhere far away.
    ///
    /// Delegates to `ring_push` so that the C ABI's caller-owned ring and this one share a
    /// single copy of that refusal. `Dim` and `Capacity` reach it as constants and fold:
    /// measured at `-Os` on x86-64, this compiles to the same 160 bytes as the hand-written
    /// loop it replaced.
    bool push(double t, const T* __restrict values, bool valid = true) {
        return ring_push<T>(v_, t_, valid_, Dim, Capacity, &head_, &count_, &dropped_,
                            &last_t_, t, values, valid);
    }

    int size() const { return count_; }

    /// Samples overwritten because the ring was full.
    ///
    /// Note what this is *not*: a stream has no idea whether anyone read a sample before it
    /// was overwritten, so this cannot mean "lost". In steady state a correctly sized ring
    /// overwrites on every push and this climbs forever with nothing wrong. The Python's
    /// docstring said "before being consumed" and was corrected for the same reason.
    ///
    /// 64 bits, not 32. `dropped + size()` is the total ever pushed, and the Python
    /// `Pipeline` derives its lost-frame count from exactly that identity; a C++ pipeline
    /// will want to as well, and an identity that stops holding after 2^32 overwrites is
    /// not one to build on. At 32 bits that is 49.7 days at 1 kHz on a device meant to run
    /// indefinitely — arithmetic, not a measurement.
    ///
    /// The widening is free here rather than four bytes: measured, `sizeof(Stream)` is
    /// unchanged at either width on x86-64 and under AAPCS, because the padding before the
    /// 8-aligned `last_t_` absorbs it. That is a host measurement of a layout rule both
    /// ABIs share, not a target measurement.
    uint64_t dropped() const { return dropped_; }

    double latest_t() const { return count_ ? last_t_ : -__builtin_inf(); }

    StreamView<T> view() const { return ring_view<T>(ring()); }

    /// This ring's storage and bookkeeping as a runtime descriptor. The same object the C
    /// ABI hands around, so the two paths cannot diverge on what a ring *is*.
    Ring<T> ring() const {
        Ring<T> r;
        r.values = const_cast<T*>(v_);
        r.times = const_cast<double*>(t_);
        r.valid = const_cast<uint8_t*>(valid_);
        r.dim = Dim;
        r.capacity = Capacity;
        r.head = head_;
        r.count = count_;
        r.dropped = dropped_;
        r.last_t = last_t_;
        return r;
    }

    SampleRef<T> at(int age) const { return view().at(age); }
    SampleRef<T> latest() const { return count_ ? at(count_ - 1) : SampleRef<T>(); }

private:
    T v_[Capacity * Dim];
    double t_[Capacity];
    uint8_t valid_[Capacity];
    int head_ = 0;
    int count_ = 0;
    uint64_t dropped_ = 0;
    double last_t_ = 0.0;
};

/// `payload_bytes` is a lower bound on the object, never a substitute for `sizeof`.
static_assert(sizeof(Stream<float, 6, 64>) >= Stream<float, 6, 64>::payload_bytes, "");
static_assert(sizeof(Stream<double, 1, 8>) >= Stream<double, 1, 8>::payload_bytes, "");

}  // namespace argmind

#endif  // ARGMIND_STREAM_HPP
