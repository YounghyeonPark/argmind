// Export a learned model on one machine, import it on another.
//
//   braincheck write <file>   train, pack the state into a record, write it
//   braincheck read  <file>   read the record, load it, predict, print
//
// The two halves are run on DIFFERENT machines -- the host, QEMU's Cortex-M4F board and
// its Cortex-M55 board -- and what has to hold is that the prediction after importing
// equals the one the exporting machine made. Otherwise a device that survives a hardware
// swap has quietly changed its mind, which is worse than refusing to start.
//
// The record format already claimed "the writer and the reader may be different
// machines" and had been checked host-against-M4F. Board-against-board is the case an
// integrator actually performs and it had never been run.
//
// The fingerprint check here is not decoration. A record whose bytes are corrupt is
// caught by the CRC; a record that is perfectly valid but was written by a device with a
// different `FrameMap` ORDER is not, and loading it would put the right numbers on the
// wrong features -- the one corruption nothing downstream can detect.

#include <stdio.h>
#include <string.h>

#include "argmind/persist.hpp"
#include "argmind/rls.hpp"

namespace {

constexpr int N = 6;
constexpr int M = 1;
constexpr uint16_t kVersion = 1;
constexpr uint32_t kFingerprint = 0xFEEDBEEFu;   // stands in for the FrameMap's

using Rls = argmind::Rls<float, N, M>;

void sample(int k, float* x, float* y) {
    for (int i = 0; i < N; ++i) x[i] = static_cast<float>((k * 7 + i * 13) % 11) - 5.0f;
    x[N - 1] = 1.0f;
    y[0] = 2.0f * x[0] - 1.5f * x[1] + 0.25f * x[2] + 0.5f;
}

void probe(float* x) {
    for (int i = 0; i < N; ++i) x[i] = static_cast<float>((4001 * 7 + i * 13) % 11) - 5.0f;
    x[N - 1] = 1.0f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: brain <write|read> <file>\n");
        return 1;
    }
    const bool writing = argv[1][0] == 'w';
    static uint8_t buf[1024];

    if (writing) {
        Rls rls(0.999f, 100.0f);
        float x[N], y[M];
        for (int k = 0; k < 3000; ++k) {
            sample(k, x, y);
            rls.update(x, y);
        }
        probe(x);
        float out[M];
        rls.predict(x, out);
        printf("exported prediction %.9f\n", static_cast<double>(out[0]));

        const size_t n = argmind::persist::pack<float>(
            rls.state(), Rls::state_size, N, M, kVersion, kFingerprint,
            /*frozen=*/false, /*cause=*/nullptr, /*cause_len=*/0, buf, sizeof(buf));
        if (n == 0) {
            fprintf(stderr, "pack refused\n");
            return 2;
        }
        FILE* f = fopen(argv[2], "wb");
        if (!f) return 3;
        const size_t w = fwrite(buf, 1, n, f);
        fclose(f);
        printf("record %u bytes\n", static_cast<unsigned>(n));
        return w == n ? 0 : 3;
    }

    FILE* f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 4;
    }
    const size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);

    argmind::persist::Record<float> rec;
    const argmind::persist::Status st = argmind::persist::unpack<float>(buf, n, &rec);
    if (st != argmind::persist::Status::Ok) {
        fprintf(stderr, "the record was refused: status %d\n", static_cast<int>(st));
        return 5;
    }
    printf("record ok: v%u %ux%u fingerprint %08x count %u\n", rec.version,
           rec.n_features, rec.n_outputs, rec.fingerprint, rec.count);
    if (rec.fingerprint != kFingerprint) {
        fprintf(stderr, "fingerprint mismatch -- refusing\n");
        return 6;
    }

    float state[Rls::state_size];
    if (!argmind::persist::read_state<float>(rec, state, Rls::state_size)) {
        fprintf(stderr, "read_state refused\n");
        return 7;
    }
    Rls rls(0.999f, 100.0f);
    if (!rls.load_state(state)) {
        fprintf(stderr, "load_state refused\n");
        return 8;
    }
    float x[N], out[M];
    probe(x);
    rls.predict(x, out);
    printf("imported prediction %.9f\n", static_cast<double>(out[0]));
    return 0;
}
