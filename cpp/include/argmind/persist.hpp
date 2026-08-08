// SPDX-License-Identifier: Apache-2.0
//
// The persistence record format, ported from argmind/persist.py.
//
// This is the piece the format was designed for. The Python reference writes files but
// models the raw-flash scheme faithfully — two slots, written alternately, each carrying
// a sequence number and a CRC — precisely because this side has no `os.replace` and a
// design that leaned on rename would not have survived the port.
//
// What is here and what is not
// ----------------------------
// This header does the **format**: pack a state block into a caller's buffer, validate and
// unpack one, and arbitrate between two slots. It does **not** write to flash. Erase
// granularity, write width, wear levelling and the driver API are all platform-specific,
// and a library that guessed at them would be wrong everywhere. The caller writes the
// bytes; this decides what the bytes are and which slot to believe.
//
// Byte-for-byte identical to the Python
// -------------------------------------
// Unlike the recursion, serialisation has no rounding, so parity here is not "to the
// tolerance the precision allows" — it is exact, and the tests assert exactly that.
//
// Two things that identity depends on:
//
//   - **No structs are laid over the buffer.** The header is 26 bytes and Python's
//     `struct` packs it without padding, so two of its three `uint32_t` fields land
//     unaligned: `fingerprint` at offset 18 and `count` at offset 22. (`sequence`, at
//     offset 8, happens to be aligned — an earlier version of this comment claimed all
//     three were unaligned and named offset 6, which holds the 16-bit flags. The code was
//     right and the comment was not.) A C++ struct laid over the buffer would insert
//     padding and produce a longer record, and a pointer cast onto it would be an
//     unaligned access: undefined behaviour, and a fault or a silent penalty on the target
//     depending on the instruction the compiler picks. Every field is read and written a
//     byte at a time.
//   - **CRC-32 must match zlib's.** That is the reflected IEEE 802.3 CRC: polynomial
//     0xEDB88320, initial value all ones, final inversion. Implemented bitwise rather than
//     from a 1 KB table: at 32 KB of flash the table is 3% of the budget, and persistence
//     runs at a declared cadence measured in minutes, so eight shifts per byte costs
//     nothing that matters. If a target ever needs the speed, a nibble table is 64 bytes.

#ifndef ARGMIND_PERSIST_HPP
#define ARGMIND_PERSIST_HPP

#include <stddef.h>
#include <stdint.h>

namespace argmind {
namespace persist {

/// Bytes 0..3 of every record.
static constexpr uint8_t MAGIC[4] = {'A', 'R', 'G', 'M'};
static constexpr uint16_t FORMAT_VERSION = 1;

static constexpr size_t HEADER_BYTES = 26;
static constexpr size_t CRC_BYTES = 4;
static constexpr size_t MAX_CAUSE_BYTES = 255;

static constexpr uint16_t FLAG_FROZEN = 1u << 0;

/// Why a slot could not be decoded. Every one of these is a failure a device actually
/// produces: a slot never written, a write cut short by power loss, a bit flipped in
/// storage, or a record written by a future version of this library.
enum class Status : uint8_t {
    Ok = 0,
    TooShort = 1,        ///< fewer bytes than a record can occupy
    BadMagic = 2,        ///< not an argmind record at all
    FutureVersion = 3,   ///< newer than this build understands; refuse rather than guess
    BadDtype = 4,
    LengthMismatch = 5,  ///< the header describes a different size: a torn write
    CrcMismatch = 6,     ///< corrupt
};

/// A decoded slot. `state` and `cause` point into the caller's buffer and are valid only
/// as long as it is; nothing here copies, because there is nowhere to copy to.
template <typename T>
struct Record {
    uint16_t version = 0;
    uint32_t sequence = 0;
    uint16_t n_features = 0;
    uint16_t n_outputs = 0;
    uint32_t fingerprint = 0;
    bool frozen = false;
    const char* cause = nullptr;   ///< not NUL-terminated; `cause_len` bytes
    uint8_t cause_len = 0;
    const uint8_t* state = nullptr;  ///< little-endian T, `count` of them
    uint32_t count = 0;
};

// -- byte-wise access, because nothing here may be pointer-cast onto the buffer ---------

inline uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1] << 8);
}

inline uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline void write_u16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
}

inline void write_u32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

/// True on a big-endian host. The record is little-endian by declaration, so this decides
/// whether the element bytes need reversing.
///
/// Checked on this host: gcc 16.1 x86-64 at `-Os` folds it away, emitting one move per
/// element with no branch and no reversal path. That is a host result. No target compiler
/// exists on this machine, so whether an M4F build does the same is expected and
/// unverified — the same rule that keeps on-target timing claims absent.
inline bool is_big_endian() {
    const uint16_t probe = 1;
    uint8_t first;
    __builtin_memcpy(&first, &probe, 1);
    return first == 0;
}

/// CRC-32 as zlib computes it. Reflected, polynomial 0xEDB88320, initial value all ones,
/// final inversion. Bitwise on purpose — see the header comment.
inline uint32_t crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

/// The dtype code stored in the header: the size of one element in bytes.
template <typename T>
constexpr uint8_t dtype_code() {
    return static_cast<uint8_t>(sizeof(T));
}

/// Exactly how many bytes a record occupies. A caller sizes its slot with this.
constexpr size_t record_bytes(size_t element_count, size_t element_size,
                              size_t cause_len = 0) {
    return HEADER_BYTES + cause_len + element_count * element_size + CRC_BYTES;
}

// -- packing ---------------------------------------------------------------------------

/// Serialise a state block into `out`. Returns the number of bytes written, or 0 if the
/// buffer is too small or the arguments are out of range.
///
/// `cause` is the guard's reason for freezing, carried alongside the block because it is a
/// conclusion rather than working state. It is truncated at 255 bytes, matching the
/// reference; the length is one byte in the header and there is nothing to gain from
/// widening it.
template <typename T>
size_t pack(const T* __restrict state, uint32_t count, uint16_t n_features,
            uint16_t n_outputs, uint32_t sequence, uint32_t fingerprint, bool frozen,
            const char* __restrict cause, size_t cause_len, uint8_t* __restrict out,
            size_t out_bytes) {
    if (!state || !out) return 0;
    if (cause_len > MAX_CAUSE_BYTES) cause_len = MAX_CAUSE_BYTES;
    if (!cause) cause_len = 0;

    // The same wraparound `unpack` guards against, on the way out. `count` is the
    // caller's here rather than untrusted input, but `size_t` is still 32 bits on the
    // target: without this, count = 0x20000000 at T=double wraps `total` to 30, passes the
    // buffer check, and then writes half a gigabyte past the caller's buffer. Guarding one
    // direction and not the other was an omission, not a decision.
    //
    // Why no test reaches this, stated as arithmetic rather than as a shrug: the guard
    // fires above `(SIZE_MAX - 285) / sizeof(T)`. On a 64-bit host that is 2.3e18 for
    // double, and `count` is a uint32_t, so no record can express a value that reaches it
    // — removing the guard is an *equivalent mutant* here, not an untested one. On the
    // 32-bit target the same expression is 536,870,876, comfortably inside uint32 range,
    // which is where it does its work.
    if (count > (SIZE_MAX - HEADER_BYTES - CRC_BYTES - cause_len) / sizeof(T)) return 0;

    const size_t total = record_bytes(count, sizeof(T), cause_len);
    if (out_bytes < total) return 0;

    out[0] = MAGIC[0];
    out[1] = MAGIC[1];
    out[2] = MAGIC[2];
    out[3] = MAGIC[3];
    write_u16(out + 4, FORMAT_VERSION);
    write_u16(out + 6, frozen ? FLAG_FROZEN : 0u);
    write_u32(out + 8, sequence);
    write_u16(out + 12, n_features);
    write_u16(out + 14, n_outputs);
    out[16] = dtype_code<T>();
    out[17] = static_cast<uint8_t>(cause_len);
    write_u32(out + 18, fingerprint);
    write_u32(out + 22, count);

    size_t at = HEADER_BYTES;
    for (size_t i = 0; i < cause_len; ++i) out[at + i] = static_cast<uint8_t>(cause[i]);
    at += cause_len;

    // Element-wise little-endian rather than a memcpy of the array: a big-endian target
    // would otherwise write a record this library could not read back.
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t bytes[sizeof(T)];
        __builtin_memcpy(bytes, &state[i], sizeof(T));
        if (is_big_endian()) {
            for (size_t b = 0; b < sizeof(T); ++b) out[at + b] = bytes[sizeof(T) - 1 - b];
        } else {
            for (size_t b = 0; b < sizeof(T); ++b) out[at + b] = bytes[b];
        }
        at += sizeof(T);
    }

    write_u32(out + at, crc32(out, at));
    return total;
}

// -- unpacking -------------------------------------------------------------------------

/// Decode one slot. Nothing is copied; `rec` points into `in`.
template <typename T>
Status unpack(const uint8_t* __restrict in, size_t in_bytes, Record<T>* __restrict rec) {
    if (!in || !rec) return Status::TooShort;
    if (in_bytes < HEADER_BYTES + CRC_BYTES) return Status::TooShort;
    if (in[0] != MAGIC[0] || in[1] != MAGIC[1] || in[2] != MAGIC[2] || in[3] != MAGIC[3])
        return Status::BadMagic;

    const uint16_t version = read_u16(in + 4);
    if (version > FORMAT_VERSION) return Status::FutureVersion;
    const uint8_t code = in[16];
    if (code != dtype_code<T>()) return Status::BadDtype;

    const uint8_t cause_len = in[17];
    const uint32_t count = read_u32(in + 22);

    // `count` comes straight out of the record, so a flipped bit can make it enormous —
    // and `size_t` is 32 bits on the Cortex-M4F floor, where `count * sizeof(T)` then
    // wraps. A forged count can land on a product that makes `expect` equal the real
    // length, so the length check alone would pass it.
    //
    // In practice the CRC still catches it, because the corrupted bytes sit inside the
    // span the CRC covers. But that makes the safety depend on an unstated invariant
    // between two independent checks, and an explicit guard costs one comparison. The
    // project's rule that host numbers are not target numbers applies to behaviour too.    //
    // Why no test reaches this, stated as arithmetic rather than as a shrug: the guard
    // fires above `(SIZE_MAX - 285) / sizeof(T)`. On a 64-bit host that is 2.3e18 for
    // double, and `count` is a uint32_t, so no record can express a value that reaches it
    // — removing the guard is an *equivalent mutant* here, not an untested one. On the
    // 32-bit target the same expression is 536,870,876, comfortably inside uint32 range,
    // which is where it does its work.
    if (count > (SIZE_MAX - HEADER_BYTES - CRC_BYTES - cause_len) / sizeof(T))
        return Status::LengthMismatch;

    const size_t expect = record_bytes(count, sizeof(T), cause_len);
    if (in_bytes != expect) return Status::LengthMismatch;

    const uint32_t stored = read_u32(in + expect - CRC_BYTES);
    if (stored != crc32(in, expect - CRC_BYTES)) return Status::CrcMismatch;

    rec->version = version;
    rec->sequence = read_u32(in + 8);
    rec->n_features = read_u16(in + 12);
    rec->n_outputs = read_u16(in + 14);
    rec->fingerprint = read_u32(in + 18);
    rec->frozen = (read_u16(in + 6) & FLAG_FROZEN) != 0;
    rec->cause = cause_len ? reinterpret_cast<const char*>(in + HEADER_BYTES) : nullptr;
    rec->cause_len = cause_len;
    rec->state = in + HEADER_BYTES + cause_len;
    rec->count = count;
    return Status::Ok;
}

/// Copy a record's state into a typed array, reversing bytes if the host demands it.
///
/// Separate from `unpack` because a caller that only wants the sequence number or the
/// frozen flag should not pay for the copy. It deliberately does **not** check the values
/// for finiteness: `Rls::load_state` already refuses a non-finite block, and a refusal
/// implemented in two places is a refusal that will eventually disagree with itself.
template <typename T>
bool read_state(const Record<T>& rec, T* __restrict out, uint32_t out_count) {
    if (!rec.state || out_count != rec.count) return false;
    for (uint32_t i = 0; i < rec.count; ++i) {
        uint8_t bytes[sizeof(T)];
        const uint8_t* src = rec.state + i * sizeof(T);
        if (is_big_endian()) {
            for (size_t b = 0; b < sizeof(T); ++b) bytes[b] = src[sizeof(T) - 1 - b];
        } else {
            for (size_t b = 0; b < sizeof(T); ++b) bytes[b] = src[b];
        }
        __builtin_memcpy(&out[i], bytes, sizeof(T));
    }
    return true;
}

// -- slot arbitration ------------------------------------------------------------------

/// Which of two slots to believe, or -1 if neither validates.
///
/// The highest sequence number that decodes wins. This is the whole point of two slots: a
/// write interrupted by power loss can destroy at most the slot being written, which is
/// never the one currently in use, so the previous record survives untouched.
///
/// `status_a` and `status_b`, when given, receive why each slot was rejected. They come
/// back rather than being dropped because "it started from scratch" and "it started from
/// scratch because slot 1 had a CRC mismatch" are very different mornings.
template <typename T>
int select_newest(const uint8_t* __restrict a, size_t a_bytes,
                  const uint8_t* __restrict b, size_t b_bytes,
                  Record<T>* __restrict out, Status* status_a = nullptr,
                  Status* status_b = nullptr) {
    Record<T> ra;
    Record<T> rb;
    const Status sa = a ? unpack<T>(a, a_bytes, &ra) : Status::TooShort;
    const Status sb = b ? unpack<T>(b, b_bytes, &rb) : Status::TooShort;
    if (status_a) *status_a = sa;
    if (status_b) *status_b = sb;

    const bool ok_a = sa == Status::Ok;
    const bool ok_b = sb == Status::Ok;
    if (!ok_a && !ok_b) return -1;
    if (ok_a && (!ok_b || ra.sequence >= rb.sequence)) {
        if (out) *out = ra;
        return 0;
    }
    if (out) *out = rb;
    return 1;
}

/// The slot the next write should go to, given which one is currently newest.
/// `newest` is the return of `select_newest`; -1 (nothing valid) starts at slot 0.
inline int next_slot(int newest) { return newest == 0 ? 1 : 0; }

}  // namespace persist
}  // namespace argmind

#endif  // ARGMIND_PERSIST_HPP
