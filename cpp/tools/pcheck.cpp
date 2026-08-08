// SPDX-License-Identifier: Apache-2.0
//
// The persistence parity tool. Packs a state block with the C++ implementation and writes
// the bytes out, so the Python can compare them against its own.
//
// Unlike the recursion, serialisation has no rounding. The claim here is **byte identity**,
// not agreement to a tolerance, and this tool exists so that claim can be checked rather
// than asserted.
//
// It also answers the questions in the other direction: given bytes the Python produced,
// does the C++ decode them to the same fields, and does it reject the same corruptions for
// the same reasons.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "argmind/persist.hpp"

namespace {

// The input describes what to pack; its own layout is chosen for easy reading rather than
// for compactness, and is naturally aligned throughout.
struct Job {
    char magic[4];  // "AMPC"
    uint16_t version;
    uint8_t dtype;  // 4 or 8
    uint8_t mode;   // 0 = pack, 1 = unpack, 2 = arbitrate, 3 = batch unpack,
                    // 4 = batch pack
    uint16_t n_features;
    uint16_t n_outputs;
    uint32_t count;
    uint32_t sequence;
    uint32_t fingerprint;
    uint32_t frozen;
    uint32_t cause_len;
};
static_assert(sizeof(Job) == 32, "job header must be 32 bytes");

// What `unpack` decided, for the Python to compare against its own decode.
struct Decoded {
    int32_t status;
    uint32_t sequence;
    uint32_t fingerprint;
    uint32_t frozen;
    uint32_t n_features;
    uint32_t n_outputs;
    uint32_t count;
    uint32_t cause_len;
};
static_assert(sizeof(Decoded) == 32, "decoded record must be 32 bytes");

// What `select_newest` decided. It exists because a mutation check found the function was
// compiled but never called from any binary the suite built: the arbitration test decoded
// each slot separately and compared against Python's own arbitration, which is a different
// code path. Two mutants — picking the lower sequence, and preferring an invalid slot —
// survived for that reason alone.
struct Arbitration {
    int32_t winner;     // 0, 1, or -1 if neither validates
    int32_t status_a;
    int32_t status_b;
    uint32_t sequence;  // of the winner, or 0
    int32_t next;       // where the following write belongs
    int32_t pad;
};
static_assert(sizeof(Arbitration) == 24, "arbitration result must be 24 bytes");

constexpr size_t kMaxBytes = 1 << 16;
// Big enough for any cause a job can reasonably carry. The point of *not* clamping to 255
// here is that the library's own clamp is then the thing under test; a tool that
// pre-clamped made that clamp unreachable and a mutant removing it survived.
constexpr size_t kMaxCause = 1 << 13;
uint8_t buffer[kMaxBytes];
// `scratch` is cast to float*/double* below, so it must be aligned for them. A bare
// uint8_t array is only guaranteed 1-byte aligned; -Wcast-align flags the cast, and it is
// undefined behaviour even where a given toolchain happens to place the array well. This
// tool exists to demonstrate that persist.hpp's alignment discipline is right, so it had
// better not violate it itself.
alignas(alignof(double)) uint8_t scratch[kMaxBytes];

template <typename T>
int do_pack(FILE* in, FILE* out, const Job& job) {
    // Bounded by the buffer, not by the library's 255 -- reading `cause_len` straight into
    // a 256-byte buffer was an out-of-bounds stack write (silent at 257, an access
    // violation at 300), but clamping to 255 *here* made the library's own clamp
    // unreachable and a mutant removing it survived. So: refuse what will not fit, and
    // otherwise hand `pack` the true length and let it do the clamping it claims to.
    static char cause[kMaxCause];
    if (job.cause_len > kMaxCause) {
        fprintf(stderr, "pcheck: cause_len %u exceeds this tool's buffer\n",
                job.cause_len);
        return 8;
    }
    const uint32_t want = job.cause_len;
    if (want) {
        if (fread(cause, 1, want, in) != want) return 4;
    }
    if (job.count * sizeof(T) > kMaxBytes) return 5;
    T* state = reinterpret_cast<T*>(scratch);
    if (fread(state, sizeof(T), job.count, in) != job.count) return 4;

    const size_t n = argmind::persist::pack<T>(
        state, job.count, job.n_features, job.n_outputs, job.sequence, job.fingerprint,
        job.frozen != 0, want ? cause : nullptr, want, buffer, sizeof(buffer));
    if (n == 0) {
        fprintf(stderr, "pcheck: pack refused the arguments\n");
        return 6;
    }
    return fwrite(buffer, 1, n, out) == n ? 0 : 3;
}

/// Decode many blobs in one invocation.
///
/// The sweeps that make the byte-identity claim reproducible — every single-bit flip in a
/// record, every truncation length, and a differential fuzz run — are thousands of decodes.
/// One process per decode put the persistence parity file at 53 seconds, essentially all of
/// it spawn cost, and a pre-commit hook that slow is a hook people learn to bypass. Batching
/// keeps the coverage and gives back the time.
///
/// Input after the job header: u32 count, then count times (u32 length, length bytes).
/// Output: count Decoded structs, in order.
template <typename T>
int do_batch_unpack(FILE* in, FILE* out, const Job& job) {
    (void)job;
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, in) != 1) return 4;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0;
        if (fread(&len, sizeof(len), 1, in) != 1) return 4;
        if (len > kMaxBytes) return 5;
        if (len && fread(buffer, 1, len, in) != len) return 4;

        argmind::persist::Record<T> rec;
        const argmind::persist::Status st =
            argmind::persist::unpack<T>(buffer, len, &rec);
        Decoded d;
        memset(&d, 0, sizeof(d));
        d.status = static_cast<int32_t>(st);
        if (st == argmind::persist::Status::Ok) {
            d.sequence = rec.sequence;
            d.fingerprint = rec.fingerprint;
            d.frozen = rec.frozen ? 1u : 0u;
            d.n_features = rec.n_features;
            d.n_outputs = rec.n_outputs;
            d.count = rec.count;
            d.cause_len = rec.cause_len;
        }
        if (fwrite(&d, sizeof(d), 1, out) != 1) return 3;
    }
    return 0;
}

/// Pack many records that share a state block but differ in their header fields.
///
/// Input after the job header: the state (job.count elements), then u32 n, then n times
/// (u32 sequence, u32 fingerprint, u32 frozen, u32 cause_len, cause bytes).
/// Output: n times (u32 length, length bytes).
template <typename T>
int do_batch_pack(FILE* in, FILE* out, const Job& job) {
    if (job.count * sizeof(T) > kMaxBytes) return 5;
    T* state = reinterpret_cast<T*>(scratch);
    if (fread(state, sizeof(T), job.count, in) != job.count) return 4;

    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, in) != 1) return 4;
    static char cause[kMaxCause];
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t sequence = 0, fingerprint = 0, frozen = 0, cause_len = 0;
        if (fread(&sequence, sizeof(sequence), 1, in) != 1) return 4;
        if (fread(&fingerprint, sizeof(fingerprint), 1, in) != 1) return 4;
        if (fread(&frozen, sizeof(frozen), 1, in) != 1) return 4;
        if (fread(&cause_len, sizeof(cause_len), 1, in) != 1) return 4;
        if (cause_len > kMaxCause) return 8;
        if (cause_len && fread(cause, 1, cause_len, in) != cause_len) return 4;

        const size_t written = argmind::persist::pack<T>(
            state, job.count, job.n_features, job.n_outputs, sequence, fingerprint,
            frozen != 0, cause_len ? cause : nullptr, cause_len, buffer, sizeof(buffer));
        if (written == 0) return 6;
        const uint32_t len = static_cast<uint32_t>(written);
        if (fwrite(&len, sizeof(len), 1, out) != 1) return 3;
        if (fwrite(buffer, 1, written, out) != written) return 3;
    }
    return 0;
}

/// Arbitrate between two slots, which is the whole reason there are two.
template <typename T>
int do_select(FILE* in, FILE* out, const Job& job) {
    (void)job;
    static uint8_t slot_a[kMaxBytes];
    static uint8_t slot_b[kMaxBytes];
    uint32_t len_a = 0;
    uint32_t len_b = 0;
    if (fread(&len_a, sizeof(len_a), 1, in) != 1) return 4;
    if (len_a > kMaxBytes) return 5;
    if (len_a && fread(slot_a, 1, len_a, in) != len_a) return 4;
    if (fread(&len_b, sizeof(len_b), 1, in) != 1) return 4;
    if (len_b > kMaxBytes) return 5;
    if (len_b && fread(slot_b, 1, len_b, in) != len_b) return 4;

    argmind::persist::Record<T> rec;
    argmind::persist::Status sa = argmind::persist::Status::TooShort;
    argmind::persist::Status sb = argmind::persist::Status::TooShort;
    const int winner = argmind::persist::select_newest<T>(
        len_a ? slot_a : nullptr, len_a, len_b ? slot_b : nullptr, len_b, &rec, &sa, &sb);

    Arbitration r;
    r.winner = winner;
    r.status_a = static_cast<int32_t>(sa);
    r.status_b = static_cast<int32_t>(sb);
    r.sequence = winner >= 0 ? rec.sequence : 0u;
    // Reported so that `next_slot` has a caller. It had none anywhere -- not in the
    // library, not in a test, not in this tool -- which is the same gap that left
    // `select_newest` compiled but never run.
    r.next = argmind::persist::next_slot(winner);
    r.pad = 0;
    return fwrite(&r, sizeof(r), 1, out) == 1 ? 0 : 3;
}

template <typename T>
int do_unpack(FILE* in, FILE* out, const Job& job) {
    const size_t n = fread(buffer, 1, sizeof(buffer), in);
    argmind::persist::Record<T> rec;
    const argmind::persist::Status st = argmind::persist::unpack<T>(buffer, n, &rec);

    Decoded d;
    memset(&d, 0, sizeof(d));
    d.status = static_cast<int32_t>(st);
    if (st == argmind::persist::Status::Ok) {
        d.sequence = rec.sequence;
        d.fingerprint = rec.fingerprint;
        d.frozen = rec.frozen ? 1u : 0u;
        d.n_features = rec.n_features;
        d.n_outputs = rec.n_outputs;
        d.count = rec.count;
        d.cause_len = rec.cause_len;
    }
    if (fwrite(&d, sizeof(d), 1, out) != 1) return 3;
    if (st != argmind::persist::Status::Ok) return 0;

    // The cause bytes, then the state, so the Python can check the payload survived and
    // not merely the header.
    if (rec.cause_len &&
        fwrite(rec.cause, 1, rec.cause_len, out) != rec.cause_len) return 3;
    if (job.count * sizeof(T) > kMaxBytes) return 5;
    T* state = reinterpret_cast<T*>(scratch);
    if (!argmind::persist::read_state<T>(rec, state, rec.count)) return 7;
    return fwrite(state, sizeof(T), rec.count, out) == rec.count ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: pcheck <job-file> <out-file>\n");
        return 1;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) return 1;
    Job job;
    if (fread(&job, sizeof(job), 1, in) != 1 || job.magic[0] != 'A' ||
        job.magic[1] != 'M' || job.magic[2] != 'P' || job.magic[3] != 'C') {
        fprintf(stderr, "pcheck: not a job file\n");
        fclose(in);
        return 1;
    }
    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        fclose(in);
        return 1;
    }

    int rc;
    if (job.mode == 0) {
        rc = job.dtype == 4 ? do_pack<float>(in, out, job) : do_pack<double>(in, out, job);
    } else if (job.mode == 1) {
        rc = job.dtype == 4 ? do_unpack<float>(in, out, job)
                            : do_unpack<double>(in, out, job);
    } else if (job.mode == 2) {
        rc = job.dtype == 4 ? do_select<float>(in, out, job)
                            : do_select<double>(in, out, job);
    } else if (job.mode == 3) {
        rc = job.dtype == 4 ? do_batch_unpack<float>(in, out, job)
                            : do_batch_unpack<double>(in, out, job);
    } else if (job.mode == 4) {
        rc = job.dtype == 4 ? do_batch_pack<float>(in, out, job)
                            : do_batch_pack<double>(in, out, job);
    } else {
        fprintf(stderr, "pcheck: unknown mode %u\n", job.mode);
        rc = 9;
    }
    fclose(in);
    fclose(out);
    return rc;
}
