// SPDX-License-Identifier: Apache-2.0
//
// Self-contained tests for the C++ core and the C ABI. No test framework: a
// dependency-free core deserves a dependency-free suite, and the whole thing is a
// few dozen assertions.
//
// Parity against the Python reference is NOT tested here — that lives in
// tests/test_cpp_parity.py, where the oracle is. What is tested here is what only
// the C++ side can check: layout, the ABI's boundary conditions, the size bounds the
// header promises, and the properties that hold regardless of the reference.

#include <stdint.h>
#include <stdio.h>

#include "argmind/argmind.h"
#include "argmind/align.hpp"
#include "argmind/guard.hpp"
#include "argmind/persist.hpp"
#include "argmind/pipeline.hpp"
#include "argmind/rls.hpp"
#include "argmind/stream.hpp"

namespace {

/// Compare two doubles by bit pattern. Across an ABI boundary the two sides are the same
/// arithmetic, so anything but bit identity is a plumbing bug and a tolerance would hide it.
inline bool bits_equal(double a, double b) {
    uint64_t ua;
    uint64_t ub;
    __builtin_memcpy(&ua, &a, sizeof(ua));
    __builtin_memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}


int failures = 0;
int checks = 0;

void check(bool ok, const char* what, int line) {
    ++checks;
    if (!ok) {
        ++failures;
        fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, line, what);
    }
}

// Variadic so that commas inside template argument lists reach the compiler
// rather than being read as extra macro arguments.
#define CHECK(...) check((__VA_ARGS__), #__VA_ARGS__, __LINE__)

bool close(double a, double b, double tol) {
    const double d = a - b;
    return (d < 0 ? -d : d) <= tol * (1.0 + (b < 0 ? -b : b));
}

// -- mat.hpp -----------------------------------------------------------------

void test_finiteness_predicates() {
    CHECK(argmind::is_finite(0.0f));
    CHECK(argmind::is_finite(-1.5f));
    CHECK(argmind::is_finite(1e38f));
    CHECK(!argmind::is_finite(__builtin_inff()));
    CHECK(!argmind::is_finite(-__builtin_inff()));
    CHECK(!argmind::is_finite(__builtin_nanf("")));
    CHECK(argmind::is_finite(1e308));
    CHECK(!argmind::is_finite(__builtin_inf()));
    CHECK(!argmind::is_finite(__builtin_nan("")));
    // Denormals are finite. A predicate that called them otherwise would reject
    // perfectly good sensor readings near zero.
    CHECK(argmind::is_finite(1e-45f));
}

void test_row_major_matches_the_declared_layout() {
    // A 2x3 matrix laid out row-major; if this ever changes, the state block no
    // longer matches numpy's and persistence breaks silently.
    double a[6] = {1, 2, 3, 4, 5, 6};
    double x[3] = {1, 10, 100};
    double out[2];
    argmind::mat_vec<double, 2, 3>(a, x, out);
    CHECK(out[0] == 1 + 20 + 300);
    CHECK(out[1] == 4 + 50 + 600);

    double xv[2] = {1, 10};
    double out2[3];
    argmind::vec_mat<double, 2, 3>(xv, a, out2);
    CHECK(out2[0] == 1 + 40);
    CHECK(out2[1] == 2 + 50);
    CHECK(out2[2] == 3 + 60);
}

void test_symmetrise_averages_pairs_and_leaves_the_diagonal() {
    double p[9] = {1, 2, 3,
                   4, 5, 6,
                   7, 8, 9};
    argmind::symmetrise<double, 3>(p);
    CHECK(p[0] == 1);          // diagonal untouched
    CHECK(p[4] == 5);
    CHECK(p[8] == 9);
    CHECK(p[1] == 3 && p[3] == 3);      // (2 + 4) / 2
    CHECK(p[2] == 5 && p[6] == 5);      // (3 + 7) / 2
    CHECK(p[5] == 7 && p[7] == 7);      // (6 + 8) / 2
}

void test_trace_accumulates_in_double_even_at_float_storage() {
    // Deliberate: the Python computes trace(P.astype(float64)). Summing many
    // float32 values in float32 would disagree with it for reasons unrelated to
    // the recursion. 2049 is chosen so the float32 sum visibly loses a term.
    const int n = 3;
    float p[n * n] = {0};
    p[0] = 1.0f;
    p[4] = 1.0f / 3.0f;
    p[8] = 1.0f / 7.0f;
    const double t = argmind::trace_f64<float, n>(p);
    const double expect = 1.0 + static_cast<double>(1.0f / 3.0f) +
                          static_cast<double>(1.0f / 7.0f);
    CHECK(t == expect);
}

// -- Rls ---------------------------------------------------------------------

void test_state_is_one_contiguous_block_theta_then_p() {
    argmind::Rls<double, 3, 2> rls(1.0, 5.0);
    CHECK(rls.state_size == 3 * 2 + 3 * 3);
    CHECK(rls.theta() == rls.state());
    CHECK(rls.p() == rls.state() + 3 * 2);
    // P starts at p0 * I, theta at zero.
    for (int i = 0; i < 6; ++i) CHECK(rls.state()[i] == 0.0);
    for (int i = 0; i < 3; ++i) CHECK(rls.p()[i * 3 + i] == 5.0);
    CHECK(rls.p()[1] == 0.0);
}

void test_it_recovers_known_parameters() {
    argmind::Rls<double, 3, 1> rls(1.0, 1e6);
    const double truth[3] = {1.5, -0.5, 2.0};
    // A deterministic, well-conditioned sweep; no RNG in a freestanding test. The
    // design matrix it builds has a condition number of 2.48 and a smallest
    // eigenvalue of 591, both computed rather than hoped for.
    //
    // The tolerance below is derived from those numbers, not chosen. p0 is a prior
    // covariance, so the recursion solves a ridge problem and leaves a bias of
    // order 1 / (p0 * lambda_min) = 1 / (1e6 * 591) ~ 1.7e-9. Asserting 1e-9 here
    // failed for exactly that reason, which is the same prior term the Python
    // suite pins by showing it scale as 1/p0.
    for (int i = 0; i < 60; ++i) {
        const double a = static_cast<double>((i * 7) % 11) - 5.0;
        const double b = static_cast<double>((i * 5) % 13) - 6.0;
        const double c = static_cast<double>((i * 3) % 17) - 8.0;
        const double x[3] = {a, b, c};
        const double y[1] = {a * truth[0] + b * truth[1] + c * truth[2]};
        const argmind::Report<double, 1> r = rls.update(x, y);
        CHECK(r.applied);
    }
    for (int i = 0; i < 3; ++i) CHECK(close(rls.theta()[i], truth[i], 1e-8));
    CHECK(rls.updates() == 60);
}

void test_a_posteriori_error_never_exceeds_a_priori() {
    // An exact property of the update: e_post = e * lambda / (lambda + x'Px).
    argmind::Rls<double, 4, 1> rls(0.97, 100.0);
    for (int i = 0; i < 120; ++i) {
        const double x[4] = {1.0 + (i % 5), 2.0 - (i % 3), (i % 7) - 3.0,
                             (i % 2) ? 1.0 : -1.0};
        const double y[1] = {static_cast<double>((i % 9) - 4)};
        const argmind::Report<double, 1> r = rls.update(x, y);
        CHECK(r.applied);
        CHECK(r.residual_post_norm <= r.residual_norm + 1e-12);
    }
}

void test_nonfinite_input_is_rejected_and_leaves_the_state_untouched() {
    argmind::Rls<double, 2, 1> rls(0.99, 10.0);
    const double x0[2] = {1.0, 2.0};
    const double y0[1] = {1.0};
    rls.update(x0, y0);

    double before[argmind::Rls<double, 2, 1>::state_size];
    argmind::copy<double, argmind::Rls<double, 2, 1>::state_size>(rls.state(),
                                                                 before);

    const double bad_x[2] = {__builtin_nan(""), 1.0};
    argmind::Report<double, 1> r = rls.update(bad_x, y0);
    CHECK(!r.applied);
    CHECK(r.reason == argmind::Reason::NonFiniteInput);

    const double bad_y[1] = {__builtin_inf()};
    r = rls.update(x0, bad_y);
    CHECK(!r.applied);
    CHECK(r.reason == argmind::Reason::NonFiniteInput);

    for (int i = 0; i < argmind::Rls<double, 2, 1>::state_size; ++i)
        CHECK(rls.state()[i] == before[i]);
    CHECK(rls.rejected_nonfinite() == 2);
    CHECK(rls.updates() == 1);
}

void test_an_indefinite_covariance_is_refused_not_propagated() {
    argmind::Rls<double, 2, 1> rls(0.99, 1.0);
    const double broken[6] = {0.0, 0.0, -1.0, 0.0, 0.0, -1.0};  // theta=0, P=-I
    CHECK(rls.load_state(broken));
    const double x[2] = {3.0, 0.0};
    const double y[1] = {1.0};
    const argmind::Report<double, 1> r = rls.update(x, y);
    CHECK(!r.applied);
    CHECK(r.reason == argmind::Reason::NonPositiveDenominator);
    CHECK(r.p_min_diag < 0.0);
    CHECK(rls.rejected_denominator() == 1);
}

void test_load_state_refuses_a_block_that_is_not_finite() {
    argmind::Rls<double, 2, 1> rls(0.99, 1.0);
    double good[6];
    argmind::copy<double, 6>(rls.state(), good);
    double bad[6];
    argmind::copy<double, 6>(good, bad);
    bad[0] = __builtin_nan("");
    CHECK(!rls.load_state(bad));
    for (int i = 0; i < 6; ++i) CHECK(rls.state()[i] == good[i]);
    CHECK(rls.load_state(good));
}

void test_configuration_is_reported_invalid_rather_than_thrown() {
    // No exceptions in this core, so a bad configuration produces a defined object
    // that says it is unusable. The ABI turns that into AM_ERR_CONFIG.
    CHECK(!argmind::Rls<float, 3, 1>(0.0f, 1.0f).valid());
    CHECK(!argmind::Rls<float, 3, 1>(1.5f, 1.0f).valid());
    CHECK(!argmind::Rls<float, 3, 1>(-0.5f, 1.0f).valid());
    CHECK(!argmind::Rls<float, 3, 1>(0.99f, 0.0f).valid());
    CHECK(!argmind::Rls<float, 3, 1>(0.99f, -1.0f).valid());
    CHECK(argmind::Rls<float, 3, 1>(0.99f, 1.0f).valid());
    CHECK(argmind::Rls<float, 3, 1>(1.0f, 1.0f).valid());
}

void test_memory_horizon_is_infinite_at_lambda_one() {
    CHECK(close(argmind::Rls<double, 2, 1>(0.99, 1.0).memory_horizon_samples(),
                100.0, 1e-9));
    CHECK(!argmind::is_finite(
        argmind::Rls<double, 2, 1>(1.0, 1.0).memory_horizon_samples()));
}

void test_multi_output_matches_independent_single_output_instances() {
    // Sharing one P across outputs must be an implementation detail, not a
    // behaviour change.
    argmind::Rls<double, 3, 2> both(0.98, 100.0);
    argmind::Rls<double, 3, 1> a(0.98, 100.0);
    argmind::Rls<double, 3, 1> b(0.98, 100.0);
    for (int i = 0; i < 40; ++i) {
        const double x[3] = {1.0 + (i % 4), (i % 5) - 2.0, (i % 3) - 1.0};
        const double y[2] = {static_cast<double>(i % 7),
                             static_cast<double>((i % 11) - 5)};
        both.update(x, y);
        a.update(x, &y[0]);
        b.update(x, &y[1]);
    }
    for (int i = 0; i < 3; ++i) {
        CHECK(close(both.theta()[i * 2 + 0], a.theta()[i], 1e-12));
        CHECK(close(both.theta()[i * 2 + 1], b.theta()[i], 1e-12));
    }
}

// -- the C ABI ---------------------------------------------------------------

void test_the_declared_size_bounds_are_bounds() {
    CHECK(am_rls_f32_6_1_bytes() <= AM_RLS_F32_6_1_MAX_BYTES);
    CHECK(am_rls_f32_6_3_bytes() <= AM_RLS_F32_6_3_MAX_BYTES);
    CHECK(am_rls_f32_12_1_bytes() <= AM_RLS_F32_12_1_MAX_BYTES);
    CHECK(am_rls_f64_6_1_bytes() <= AM_RLS_F64_6_1_MAX_BYTES);
    // ...and not absurdly loose, or the macro stops being useful for sizing.
    CHECK(am_rls_f32_6_1_bytes() * 2 >= AM_RLS_F32_6_1_MAX_BYTES);
    CHECK(am_rls_f32_6_1_state_size() == AM_RLS_F32_6_1_STATE);
    CHECK(am_rls_f32_6_3_state_size() == AM_RLS_F32_6_3_STATE);
    CHECK(am_rls_f64_6_1_state_size() == AM_RLS_F64_6_1_STATE);
}

void test_the_abi_rejects_null_and_undersized_storage() {
    static am_rls_storage store[AM_RLS_F32_6_1_MAX_BYTES /
                               sizeof(am_rls_storage)];
    CHECK(am_rls_f32_6_1_init(nullptr, sizeof(store), 0.999f, 100.0f) ==
          AM_ERR_NULL);
    CHECK(am_rls_f32_6_1_init(store, 8, 0.999f, 100.0f) == AM_ERR_TOO_SMALL);
    CHECK(am_rls_f32_6_1_init(store, sizeof(store), 0.0f, 100.0f) ==
          AM_ERR_CONFIG);
    CHECK(am_rls_f32_6_1_init(store, sizeof(store), 1.5f, 100.0f) ==
          AM_ERR_CONFIG);
    CHECK(am_rls_f32_6_1_init(store, sizeof(store), 0.999f, -1.0f) ==
          AM_ERR_CONFIG);
    CHECK(am_rls_f32_6_1_init(store, sizeof(store), 0.999f, 100.0f) == AM_OK);

    am_report rep;
    float x[6] = {1, 2, 3, 4, 5, 6};
    float y[1] = {1};
    CHECK(am_rls_f32_6_1_update(nullptr, x, y, &rep, nullptr) == AM_ERR_NULL);
    CHECK(am_rls_f32_6_1_update(store, nullptr, y, &rep, nullptr) == AM_ERR_NULL);
    CHECK(am_rls_f32_6_1_update(store, x, y, nullptr, nullptr) == AM_ERR_NULL);
    CHECK(am_rls_f32_6_1_state(nullptr) == nullptr);
}

void test_the_abi_carries_the_report_and_the_residual() {
    static am_rls_storage store[AM_RLS_F32_6_3_MAX_BYTES /
                                sizeof(am_rls_storage)];
    CHECK(am_rls_f32_6_3_init(store, sizeof(store), 0.999f, 100.0f) == AM_OK);

    float x[6] = {1, 0, 0, 0, 0, 1};
    float y[3] = {2, -3, 4};
    am_report rep;
    float residual[3] = {0, 0, 0};
    CHECK(am_rls_f32_6_3_update(store, x, y, &rep, residual) == AM_OK);
    CHECK(rep.applied == 1);
    CHECK(rep.reason == AM_REASON_NONE);
    // theta starts at zero, so the a priori residual is y itself.
    CHECK(residual[0] == 2.0f && residual[1] == -3.0f && residual[2] == 4.0f);
    CHECK(rep.p_trace > 0.0);
    CHECK(rep.gain_norm > 0.0);
    CHECK(am_rls_f32_6_3_updates(store) == 1u);

    float bad[6] = {__builtin_nanf(""), 0, 0, 0, 0, 0};
    CHECK(am_rls_f32_6_3_update(store, bad, y, &rep, nullptr) == AM_OK);
    CHECK(rep.applied == 0);
    CHECK(rep.reason == AM_REASON_NONFINITE_INPUT);
    CHECK(am_rls_f32_6_3_updates(store) == 1u);
}

void test_the_abi_round_trips_a_state_block() {
    static am_rls_storage a[AM_RLS_F64_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    static am_rls_storage b[AM_RLS_F64_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_rls_f64_6_1_init(a, sizeof(a), 0.99, 100.0) == AM_OK);
    CHECK(am_rls_f64_6_1_init(b, sizeof(b), 0.99, 100.0) == AM_OK);

    am_report rep;
    for (int i = 0; i < 20; ++i) {
        double x[6] = {1.0 + i, 2.0 - i, (i % 3) - 1.0, 0.5, -0.25, 1.0};
        double y[1] = {static_cast<double>(i % 5)};
        CHECK(am_rls_f64_6_1_update(a, x, y, &rep, nullptr) == AM_OK);
    }
    CHECK(am_rls_f64_6_1_load_state(b, am_rls_f64_6_1_state(a)) == AM_OK);
    for (int i = 0; i < AM_RLS_F64_6_1_STATE; ++i)
        CHECK(am_rls_f64_6_1_state(a)[i] == am_rls_f64_6_1_state(b)[i]);

    double corrupt[AM_RLS_F64_6_1_STATE];
    for (int i = 0; i < AM_RLS_F64_6_1_STATE; ++i)
        corrupt[i] = am_rls_f64_6_1_state(a)[i];
    corrupt[0] = __builtin_nan("");
    CHECK(am_rls_f64_6_1_load_state(b, corrupt) == AM_ERR_STATE);
    // ...and the refused load changed nothing.
    CHECK(am_rls_f64_6_1_state(b)[0] == am_rls_f64_6_1_state(a)[0]);
}

void test_reset_returns_to_the_construction_state() {
    static am_rls_storage store[AM_RLS_F32_6_1_MAX_BYTES /
                                sizeof(am_rls_storage)];
    CHECK(am_rls_f32_6_1_init(store, sizeof(store), 0.99f, 7.0f) == AM_OK);
    float fresh[AM_RLS_F32_6_1_STATE];
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i)
        fresh[i] = am_rls_f32_6_1_state(store)[i];

    am_report rep;
    float x[6] = {1, 1, 1, 1, 1, 1};
    float y[1] = {3};
    for (int i = 0; i < 5; ++i) am_rls_f32_6_1_update(store, x, y, &rep, nullptr);
    bool moved = false;
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i)
        if (am_rls_f32_6_1_state(store)[i] != fresh[i]) moved = true;
    CHECK(moved);

    CHECK(am_rls_f32_6_1_reset(store) == AM_OK);
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i)
        CHECK(am_rls_f32_6_1_state(store)[i] == fresh[i]);
}


// -- Guard -------------------------------------------------------------------
//
// These exist because a mutation check found eight survivors in the guarded parity
// suite, and several of them were unreachable from the parity tool by construction: it
// never builds an invalid guard, never calls load_state on one, and never produces a
// non-finite report scalar. A harness can only test what it exercises, so the checks
// that only the C++ side can reach belong here.

using GuardedD = argmind::Guard<argmind::Rls<double, 3, 1>, 4, 4>;

GuardedD make_guard(double hi = 1e9) {
    GuardedD g(argmind::Rls<double, 3, 1>(0.99, 100.0));
    g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, hi));
    return g;
}

void test_bound_treats_non_finite_as_unhealthy() {
    // A NaN compares false against everything, so without saying so explicitly it slips
    // through every comparison as "not out of range". Unreachable from the parity tool,
    // which never produces a non-finite report scalar.
    argmind::GuardReport<double, 1> r;
    r.p_trace = 1.0;
    const argmind::Bound b = argmind::Bound::between(argmind::Field::PTrace, 0.0, 1e6);
    CHECK(b.evaluate(r));
    r.p_trace = __builtin_nan("");
    CHECK(!b.evaluate(r));
    r.p_trace = __builtin_inf();
    CHECK(!b.evaluate(r));
    r.p_trace = -__builtin_inf();
    CHECK(!b.evaluate(r));
}

void test_a_bound_that_cannot_fail_is_invalid() {
    argmind::Bound none;                       // neither limit set
    CHECK(!none.valid());
    CHECK(argmind::Bound::at_least(argmind::Field::PMinDiag, 0.0).valid());
    CHECK(argmind::Bound::at_most(argmind::Field::PTrace, 1.0).valid());
    argmind::Bound inverted =
        argmind::Bound::between(argmind::Field::PTrace, 10.0, 1.0);
    CHECK(!inverted.valid());
}

void test_a_guard_with_no_checks_is_invalid() {
    // A guard that can never freeze is a wrapper that looks like a safety mechanism.
    GuardedD bare(argmind::Rls<double, 3, 1>(0.99, 100.0));
    CHECK(!bare.valid());
    CHECK(make_guard().valid());
}

void test_a_recovery_configuration_that_could_never_recover_is_invalid() {
    // gain_norm is identically zero while frozen, so a positive lower bound on it can
    // never be satisfied and the guard would never thaw.
    GuardedD deadlocked(argmind::Rls<double, 3, 1>(0.99, 100.0), 1, 10);
    CHECK(deadlocked.add_bound(
        argmind::Bound::at_least(argmind::Field::GainNorm, 0.1)));
    CHECK(!deadlocked.valid());

    // ...and the same check is fine without automatic recovery.
    GuardedD manual_only(argmind::Rls<double, 3, 1>(0.99, 100.0), 1, 0);
    CHECK(manual_only.add_bound(
        argmind::Bound::at_least(argmind::Field::GainNorm, 0.1)));
    CHECK(manual_only.valid());
}

void test_an_invalid_divergence_configuration_is_invalid() {
    GuardedD g = make_guard();
    argmind::Divergence d;
    d.ratio = 1.0;                             // at or below 1 fires on ordinary noise
    g.set_divergence(d);
    CHECK(!g.valid());

    argmind::Divergence slower_fast;
    slower_fast.fast_alpha = 0.01;
    slower_fast.slow_alpha = 0.5;              // the "fast" average is the slower one
    g.set_divergence(slower_fast);
    CHECK(!g.valid());

    argmind::Divergence good;
    g.set_divergence(good);
    CHECK(g.valid());
}

void test_bounds_beyond_capacity_are_refused_rather_than_dropped() {
    GuardedD g(argmind::Rls<double, 3, 1>(0.99, 100.0));
    for (int i = 0; i < 4; ++i) {
        CHECK(g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, 1e9)));
    }
    CHECK(!g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, 1e9)));
    CHECK(!g.add_bound(argmind::Bound()));      // and an unusable one is refused too
}

void test_load_state_re_arms_the_guard() {
    // Every checkpoint describes a state that no longer exists, so keeping them would let
    // a later trip roll "back" to a state that was never reached. Unreachable from the
    // parity tool, which never loads state into a guard.
    GuardedD g = make_guard(1.0);              // trips immediately
    const double x[3] = {1.0, 2.0, 3.0};
    const double y[1] = {1.0};
    g.update(x, y);
    CHECK(g.frozen());

    argmind::Rls<double, 3, 1> fresh(0.99, 100.0);
    CHECK(g.load_state(fresh.state()));
    CHECK(!g.frozen());
    for (int i = 0; i < GuardedD::state_size; ++i)
        CHECK(g.state()[i] == fresh.state()[i]);

    // A refused load must not re-arm either, since nothing moved.
    double corrupt[GuardedD::state_size];
    argmind::copy<double, GuardedD::state_size>(fresh.state(), corrupt);
    corrupt[0] = __builtin_nan("");
    CHECK(!g.load_state(corrupt));
    CHECK(g.state()[0] == fresh.state()[0]);
}

void test_manual_freeze_stops_adaptation_without_rolling_back() {
    // For a supervisor that knows something the checks cannot see. Note that after a
    // manual freeze the state is NOT a checkpoint, which is the one path where the probe's
    // stored covariance scalars differ from a freshly computed pair.
    GuardedD g = make_guard();
    const double x[3] = {1.0, -2.0, 0.5};
    const double y[1] = {3.0};
    for (int i = 0; i < 10; ++i) g.update(x, y);
    double here[GuardedD::state_size];
    argmind::copy<double, GuardedD::state_size>(g.state(), here);

    g.freeze();
    CHECK(g.frozen());
    CHECK(g.rollbacks() == 0);
    for (int i = 0; i < GuardedD::state_size; ++i) CHECK(g.state()[i] == here[i]);

    const argmind::GuardReport<double, 1> probe = g.update(x, y);
    CHECK(!probe.applied);
    CHECK(probe.frozen);
    CHECK(probe.gain_norm == 0.0);
    for (int i = 0; i < GuardedD::state_size; ++i) CHECK(g.state()[i] == here[i]);
    // The covariance scalars must describe the LIVE state, not the newest checkpoint.
    // Reading the checkpoint was a bug in the Python reference, found by a mutation check
    // on this port: after a manual freeze nothing was restored, so a checkpoint can be up
    // to checkpoint_every updates stale. Here the cadence is 1, so this only confirms the
    // easy case; test_a_manual_freeze_mid_cadence_reports_the_live_covariance is the one
    // that discriminates.
    double live_trace = 0.0, live_min = 0.0;
    g.health(&live_trace, &live_min);
    CHECK(probe.p_trace == live_trace);
    CHECK(probe.p_min_diag == live_min);

    g.thaw();
    CHECK(!g.frozen());
    CHECK(g.thaws() == 1);
    CHECK(g.update(x, y).applied);
}

void test_a_manual_freeze_mid_cadence_reports_the_live_covariance() {
    // The discriminating case: three updates into a cadence of five, so the newest
    // checkpoint is the construction state and the live state is far from it. Measured in
    // the Python reference before the fix: 300.0 reported against a live 104.1.
    using G = argmind::Guard<argmind::Rls<double, 3, 1>, 4, 4>;
    G g(argmind::Rls<double, 3, 1>(0.99, 100.0), 5, 0);
    CHECK(g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, 1e9)));
    for (int i = 0; i < 3; ++i) {
        const double x[3] = {1.0 + i, -2.0, 0.5};
        const double y[1] = {3.0};
        g.update(x, y);
    }
    double live_trace = 0.0, live_min = 0.0;
    g.health(&live_trace, &live_min);
    CHECK(live_trace < 200.0);                 // well away from the 300 at construction

    g.freeze();
    const double x[3] = {1.0, -2.0, 0.5};
    const double y[1] = {3.0};
    const argmind::GuardReport<double, 1> probe = g.update(x, y);
    CHECK(probe.frozen);
    CHECK(probe.p_trace == live_trace);
    CHECK(probe.p_min_diag == live_min);
}

void test_the_checkpoint_memory_is_declared_and_matches_the_ring() {
    // Memory is a function of configuration, never of uptime.
    using G4 = argmind::Guard<argmind::Rls<float, 6, 1>, 4, 4>;
    using G1 = argmind::Guard<argmind::Rls<float, 6, 1>, 1, 4>;
    CHECK(G4::checkpoint_bytes == 4 * (42 * 4 + 16));
    CHECK(G1::checkpoint_bytes == 1 * (42 * 4 + 16));
    CHECK(G4::checkpoint_bytes == 4 * G1::checkpoint_bytes);
    CHECK(sizeof(G4) > sizeof(G1));
    CHECK(G4::state_size == argmind::Rls<float, 6, 1>::state_size);
}

void test_a_rejected_sample_is_not_a_guard_trip() {
    GuardedD g = make_guard();
    const double x[3] = {1.0, 2.0, 3.0};
    const double y[1] = {1.0};
    g.update(x, y);

    const double bad[3] = {__builtin_nan(""), 1.0, 1.0};
    const argmind::GuardReport<double, 1> r = g.update(bad, y);
    CHECK(!r.applied);
    CHECK(r.reason == argmind::Reason::NonFiniteInput);
    CHECK(!r.frozen);
    CHECK(!r.rolled_back);
    CHECK(r.tripped == 0);
    CHECK(!g.frozen());
    CHECK(g.rejected_by_adapter() == 1);
    CHECK(g.trips() == 0);
}

void test_a_second_trip_does_not_roll_forward() {
    // After a rollback the retained checkpoints are all at or after the restored state, so
    // keeping them would let the next trip restore a moment that came *afterwards*. Only
    // reachable with depth > 1 and two trips, which is why a mutation that kept them
    // survived a suite that had neither.
    //
    // Windup gives a trip at a real step rather than on the first update: excitation is
    // confined to the first two axes, so tr(P) grows as p0*lambda^-k in the third and
    // crosses the bound around step ln(5)/ln(1/0.95) = 31.
    using G = argmind::Guard<argmind::Rls<double, 3, 1>, 4, 4>;
    G g(argmind::Rls<double, 3, 1>(0.95, 1.0), 1, 0);
    CHECK(g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, 5.0)));
    CHECK(g.valid());

    int trip = -1;
    for (int k = 0; k < 200 && trip < 0; ++k) {
        const double x[3] = {1.0 + (k % 5), (k % 7) - 3.0, 0.0};
        const double y[1] = {x[0] - 2.0 * x[1]};
        if (g.update(x, y).rolled_back) trip = k;
    }
    CHECK(trip > 4);          // several checkpoints must exist behind the trip
    CHECK(g.frozen());
    double first[G::state_size];
    argmind::copy<double, G::state_size>(g.state(), first);

    g.thaw();
    int second = -1;
    for (int k = 0; k < 200 && second < 0; ++k) {
        const double x[3] = {1.0 + (k % 5), (k % 7) - 3.0, 0.0};
        const double y[1] = {x[0] - 2.0 * x[1]};
        if (g.update(x, y).rolled_back) second = k;
    }
    CHECK(second >= 0);
    CHECK(g.trips() == 2);
    // The second rollback must land on the state the first one restored, never later:
    // after a rollback the only checkpoint still in play is the one just restored.
    for (int i = 0; i < G::state_size; ++i) CHECK(g.state()[i] == first[i]);
}


// -- persist.hpp -------------------------------------------------------------
//
// These exist because an audit noticed `persist.hpp` had zero coverage in this file and
// was not in the freestanding or C99 gates either. That is the same position `guard.hpp`
// was in when eight of eighteen mutants survived: a parity harness can only test what its
// tool exercises, and the tool cannot reach a header's own edge cases.

void test_crc32_matches_the_known_vectors() {
    // The reflected IEEE 802.3 CRC, which is what zlib computes. The check vector for
    // "123456789" is 0xCBF43926 and is quoted in every catalogue of this polynomial; a
    // non-reflected implementation of the same polynomial gives a different value, and
    // that is precisely the substitution a port makes by accident.
    const char nine[] = "123456789";
    CHECK(argmind::persist::crc32(reinterpret_cast<const uint8_t*>(nine), 9) ==
          0xCBF43926u);
    CHECK(argmind::persist::crc32(nullptr, 0) == 0u);
    const uint8_t zero = 0;
    CHECK(argmind::persist::crc32(&zero, 1) == 0xD202EF8Du);
}

void test_the_byte_accessors_are_little_endian_whatever_the_host_is() {
    uint8_t buf[4] = {0, 0, 0, 0};
    argmind::persist::write_u32(buf, 0x11223344u);
    CHECK(buf[0] == 0x44 && buf[1] == 0x33 && buf[2] == 0x22 && buf[3] == 0x11);
    CHECK(argmind::persist::read_u32(buf) == 0x11223344u);

    argmind::persist::write_u16(buf, 0xBEEFu);
    CHECK(buf[0] == 0xEF && buf[1] == 0xBE);
    CHECK(argmind::persist::read_u16(buf) == 0xBEEFu);
}

void test_pack_refuses_a_buffer_that_is_too_small() {
    const double state[3] = {1.0, 2.0, 3.0};
    uint8_t out[64];
    const size_t need = argmind::persist::record_bytes(3, sizeof(double), 0);
    CHECK(need == 26 + 24 + 4);
    CHECK(argmind::persist::pack<double>(state, 3, 3, 1, 1, 0, false, nullptr, 0, out,
                                         need - 1) == 0);
    CHECK(argmind::persist::pack<double>(state, 3, 3, 1, 1, 0, false, nullptr, 0, out,
                                         need) == need);
    CHECK(argmind::persist::pack<double>(nullptr, 3, 3, 1, 1, 0, false, nullptr, 0, out,
                                         need) == 0);
    CHECK(argmind::persist::pack<double>(state, 3, 3, 1, 1, 0, false, nullptr, 0, nullptr,
                                         need) == 0);
}

void test_pack_clamps_an_overlong_cause() {
    // The length is one byte in the header, so anything past 255 is discarded. Reachable
    // only from here: the parity tool bounds its own read long before this matters.
    const float state[2] = {1.0f, 2.0f};
    static char cause[600];
    for (int i = 0; i < 600; ++i) cause[i] = 'x';
    uint8_t out[1024];
    const size_t n = argmind::persist::pack<float>(state, 2, 2, 1, 1, 0, true, cause, 600,
                                                   out, sizeof(out));
    CHECK(n == argmind::persist::record_bytes(2, sizeof(float), 255));
    CHECK(out[17] == 255);
}

void test_unpack_reports_each_failure_distinctly() {
    const float state[2] = {1.5f, -2.5f};
    uint8_t out[256];
    const size_t n = argmind::persist::pack<float>(state, 2, 2, 1, 9, 0x1234, false,
                                                   nullptr, 0, out, sizeof(out));
    CHECK(n > 0);

    argmind::persist::Record<float> rec;
    CHECK(argmind::persist::unpack<float>(out, n, &rec) == argmind::persist::Status::Ok);
    CHECK(rec.sequence == 9);
    CHECK(rec.fingerprint == 0x1234u);
    CHECK(rec.count == 2);
    CHECK(!rec.frozen);

    CHECK(argmind::persist::unpack<float>(out, 8, &rec) ==
          argmind::persist::Status::TooShort);

    uint8_t bad[256];
    argmind::copy<uint8_t, 256>(out, bad);
    bad[0] = 'X';
    CHECK(argmind::persist::unpack<float>(bad, n, &rec) ==
          argmind::persist::Status::BadMagic);

    argmind::copy<uint8_t, 256>(out, bad);
    argmind::persist::write_u16(bad + 4, argmind::persist::FORMAT_VERSION + 1);
    CHECK(argmind::persist::unpack<float>(bad, n, &rec) ==
          argmind::persist::Status::FutureVersion);

    // Note what the block above proves: the version is reported even though that edit
    // also broke the CRC. Ordering matters — a newer writer is a more useful diagnosis
    // than a corrupt slot — and a mutant moving that check after the CRC passed the whole
    // Python-side suite before this existed.

    argmind::copy<uint8_t, 256>(out, bad);
    bad[16] = 7;  // neither 4 nor 8
    CHECK(argmind::persist::unpack<float>(bad, n, &rec) ==
          argmind::persist::Status::BadDtype);

    argmind::copy<uint8_t, 256>(out, bad);
    bad[30] ^= 0x01;
    CHECK(argmind::persist::unpack<float>(bad, n, &rec) ==
          argmind::persist::Status::CrcMismatch);

    argmind::copy<uint8_t, 256>(out, bad);
    argmind::persist::write_u32(bad + 22, 99);  // a count the length cannot match
    CHECK(argmind::persist::unpack<float>(bad, n, &rec) ==
          argmind::persist::Status::LengthMismatch);
}

void test_read_state_round_trips_and_refuses_a_mismatched_count() {
    const double state[4] = {1.0, -0.0, 1e308, 4.5};
    uint8_t out[256];
    const size_t n = argmind::persist::pack<double>(state, 4, 2, 2, 1, 0, false, nullptr,
                                                    0, out, sizeof(out));
    argmind::persist::Record<double> rec;
    CHECK(argmind::persist::unpack<double>(out, n, &rec) == argmind::persist::Status::Ok);

    double back[4] = {0, 0, 0, 0};
    CHECK(argmind::persist::read_state<double>(rec, back, 4));
    for (int i = 0; i < 4; ++i) CHECK(back[i] == state[i]);
    // -0.0 must survive as -0.0, which `==` cannot tell from +0.0; check the sign bit.
    CHECK(__builtin_signbit(back[1]));

    // The rejection no binary was exercising.
    CHECK(!argmind::persist::read_state<double>(rec, back, 3));
    CHECK(!argmind::persist::read_state<double>(rec, back, 5));
}

void test_select_newest_and_next_slot() {
    const float state[2] = {1.0f, 2.0f};
    uint8_t a[256];
    uint8_t b[256];
    const size_t na = argmind::persist::pack<float>(state, 2, 2, 1, 4, 0, false, nullptr,
                                                    0, a, sizeof(a));
    const size_t nb = argmind::persist::pack<float>(state, 2, 2, 1, 9, 0, false, nullptr,
                                                    0, b, sizeof(b));
    argmind::persist::Record<float> rec;
    argmind::persist::Status sa;
    argmind::persist::Status sb;

    CHECK(argmind::persist::select_newest<float>(a, na, b, nb, &rec, &sa, &sb) == 1);
    CHECK(rec.sequence == 9);
    CHECK(sa == argmind::persist::Status::Ok && sb == argmind::persist::Status::Ok);
    CHECK(argmind::persist::next_slot(1) == 0);

    CHECK(argmind::persist::select_newest<float>(b, nb, a, na, &rec, &sa, &sb) == 0);
    CHECK(rec.sequence == 9);
    CHECK(argmind::persist::next_slot(0) == 1);

    // A torn slot must lose to an intact one whichever side it is on.
    CHECK(argmind::persist::select_newest<float>(b, 12, a, na, &rec, &sa, &sb) == 1);
    CHECK(rec.sequence == 4);
    CHECK(sa == argmind::persist::Status::TooShort);

    // Neither valid: -1, and the next write starts at slot 0.
    CHECK(argmind::persist::select_newest<float>(a, 4, b, 4, &rec, &sa, &sb) == -1);
    CHECK(argmind::persist::next_slot(-1) == 0);

    // Equal sequences resolve to slot 0, deterministically.
    CHECK(argmind::persist::select_newest<float>(b, nb, b, nb, &rec, &sa, &sb) == 0);
}


// -- stream.hpp and align.hpp -------------------------------------------------
//
// `cpp/tools/scheck.cpp` drives the aligner through a hand-built ring, because the aligner
// has to see several widths at once and a width is a template parameter. That means the
// parity tests never execute `Stream<>::push` at all — the shipped class was covered by a
// copy of itself. This is where it gets exercised, along with the things only this side can
// reach: the dimension cap, the 64-bit drop counter, and the scan that had to stay
// tie-compatible when it gained an early exit.

void test_stream_push_refuses_what_the_reference_refuses() {
    argmind::Stream<double, 2, 8> s;
    const double v[2] = {1.0, 2.0};

    CHECK(s.push(1.0, v));
    CHECK(!s.push(1.0, v));           // equal is refused
    CHECK(!s.push(0.9, v));           // backwards is refused
    CHECK(s.push(1.5, v));

    // Which check refuses which value, because they differ and an earlier version of this
    // comment got it backwards. `!(t > last_t_)` already refuses NaN and -inf; only +inf
    // reaches the finiteness test — and only that CHECK fails if the test is removed,
    // which a mutation run confirmed. The Python's hole was the other one: `t <= last` is
    // False for NaN, so a NaN was accepted and poisoned the comparison for good.
    CHECK(!s.push(__builtin_nan(""), v));       // caught by the ordering test
    CHECK(!s.push(-__builtin_inf(), v));        // likewise
    CHECK(!s.push(__builtin_inf(), v));         // ONLY the finiteness test catches this
    // ...and the reason +inf has to be refused: it passes the ordering test and then makes
    // every later push impossible, killing the stream in silence.
    CHECK(s.push(2.0, v));
    CHECK(s.push(2.5, v));            // and ordering still works afterwards
    CHECK(s.size() == 4);
}

void test_stream_overwrites_and_counts() {
    argmind::Stream<float, 1, 4> s;
    for (int i = 0; i < 10; ++i) {
        const float v[1] = {static_cast<float>(i)};
        CHECK(s.push(0.1 * (i + 1), v));
    }
    CHECK(s.size() == 4);
    CHECK(s.dropped() == 6u);
    for (int age = 0; age < 4; ++age) CHECK(s.at(age).values[0] == float(6 + age));

    // The counter is 64-bit on purpose: `dropped + size()` is the total ever pushed and the
    // pipeline derives lost frames from that identity, so a 32-bit wrap would make reported
    // loss go negative rather than merely lose a statistic. At 1 kHz that wrap is 49.7 days.
    CHECK(sizeof(s.dropped()) == 8);
}

void test_the_ring_index_survives_wrapping() {
    // The conditional-add form of `slot()` replaced two modulos; it has to agree with the
    // modulo form for every (head, count, age) a ring can reach.
    argmind::Stream<double, 1, 5> s;
    for (int i = 0; i < 23; ++i) {
        const double v[1] = {static_cast<double>(i)};
        CHECK(s.push(double(i + 1), v));
    }
    const argmind::StreamView<double> view = s.view();
    CHECK(view.count == 5);
    for (int age = 0; age < view.count; ++age) {
        const int expect = ((view.head - view.count + age) % view.capacity +
                            view.capacity) % view.capacity;
        CHECK(view.slot(age) == expect);
    }
    for (int age = 0; age < 5; ++age) CHECK(s.at(age).values[0] == double(18 + age));
}

void test_nearest_resolves_a_tie_to_the_earlier_sample() {
    argmind::Stream<double, 1, 8> s;
    for (int i = 1; i <= 5; ++i) {
        const double v[1] = {double(i)};
        CHECK(s.push(double(i), v));       // t = 1..5
    }
    const argmind::StreamView<double> v = s.view();
    CHECK(v.nearest(0.0).values[0] == 1.0);
    CHECK(v.nearest(3.2).values[0] == 3.0);
    CHECK(v.nearest(3.6).values[0] == 4.0);
    CHECK(v.nearest(99.0).values[0] == 5.0);
    // A tie must resolve to the EARLIER sample, which is what the reference's strict
    // `d < best_d` does — and it is that comparison, not the early exit, that does the
    // work. Mutating `d < best_d` to `d <= best_d` fails this line; loosening or removing
    // the break changes nothing observable, which is why this test is named for the tie
    // rather than for the exit.
    CHECK(v.nearest(3.5).values[0] == 3.0);
}

void test_bracketing_refuses_to_extrapolate() {
    argmind::Stream<double, 1, 8> s;
    const double a[1] = {10.0};
    const double b[1] = {20.0};
    CHECK(s.push(1.0, a));
    CHECK(s.push(2.0, b));
    const argmind::StreamView<double> v = s.view();

    argmind::SampleRef<double> lo;
    argmind::SampleRef<double> hi;
    CHECK(v.bracketing(1.5, &lo, &hi));
    CHECK(lo.values[0] == 10.0 && hi.values[0] == 20.0);
    CHECK(!v.bracketing(0.5, &lo, &hi));    // before the first
    CHECK(!v.bracketing(2.5, &lo, &hi));    // after the last
    CHECK(v.bracketing(2.0, &lo, &hi));     // exactly on a sample
    CHECK(lo.values[0] == 20.0 && hi.values[0] == 20.0);
}

void test_only_interpolation_is_bounded_by_the_blend_buffer() {
    // The cap exists because interpolation has to synthesise a value and there is no heap.
    // Every other policy hands back a pointer into the ring, so a wide stream must still
    // work under them — the guard used to sit ahead of the policy dispatch and refused all
    // four, which reported a wiring mistake as a dead sensor.
    constexpr int wide = argmind::Pick<double>::MAX_DIM + 1;
    static double values[2 * wide];
    static double times[2] = {1.0, 2.0};
    static uint8_t valid[2] = {1, 1};
    for (int i = 0; i < 2 * wide; ++i) values[i] = double(i);

    argmind::StreamView<double> v;
    v.times = times;
    v.values = values;
    v.valid = valid;
    v.dim = wide;
    v.capacity = 2;
    v.head = 0;
    v.count = 2;

    argmind::Pick<double> p;
    // A plain array rather than a braced-init range-for: that form needs
    // <initializer_list>, which is a hosted header this core does without.
    const argmind::Policy borrowing[3] = {argmind::Policy::HoldLast,
                                          argmind::Policy::Nearest,
                                          argmind::Policy::DropIfStale};
    for (int k = 0; k < 3; ++k) {
        const argmind::Aligner al(borrowing[k], 0.0);
        CHECK(al.pick<double>(v, 1.5, &p));
        CHECK(p.status == argmind::PickStatus::Ok);
        CHECK(p.values[0] == 0.0);
    }

    const argmind::Aligner interp(argmind::Policy::Interpolate, 0.0);
    CHECK(!interp.pick<double>(v, 1.5, &p));
    CHECK(p.status == argmind::PickStatus::DimTooLarge);
    CHECK(!p.present);

    // ...and a stream exactly at the cap still interpolates.
    v.dim = argmind::Pick<double>::MAX_DIM;
    CHECK(interp.pick<double>(v, 1.5, &p));
    CHECK(p.status == argmind::PickStatus::Ok);
}

void test_a_missing_sample_and_a_too_wide_stream_are_distinguishable() {
    // Both halves, since the name promises both. An empty ring under any policy is
    // NoSample; a stream too wide to interpolate is DimTooLarge. They used to be the same
    // return value and the same `present == false`, so a wiring mistake read as a dead
    // sensor — which is the one thing the staleness reporting exists to prevent.
    argmind::Stream<double, 1, 4> s;
    const argmind::Aligner hold(argmind::Policy::HoldLast, 0.0);
    argmind::Pick<double> p;
    CHECK(!hold.pick<double>(s.view(), 1.0, &p));
    CHECK(p.status == argmind::PickStatus::NoSample);
    CHECK(!p.present);

    constexpr int wide = argmind::Pick<double>::MAX_DIM + 1;
    static double values[2 * wide] = {0};
    static double times[2] = {1.0, 2.0};
    static uint8_t valid[2] = {1, 1};
    argmind::StreamView<double> v;
    v.times = times;
    v.values = values;
    v.valid = valid;
    v.dim = wide;
    v.capacity = 2;
    v.head = 0;
    v.count = 2;

    const argmind::Aligner interp(argmind::Policy::Interpolate, 0.0);
    CHECK(!interp.pick<double>(v, 1.5, &p));
    CHECK(p.status == argmind::PickStatus::DimTooLarge);
    CHECK(p.status != argmind::PickStatus::NoSample);
}

void test_the_causal_flag_names_the_policies_that_look_ahead() {
    CHECK(argmind::is_causal(argmind::Policy::HoldLast));
    CHECK(argmind::is_causal(argmind::Policy::DropIfStale));
    CHECK(!argmind::is_causal(argmind::Policy::Nearest));
    CHECK(!argmind::is_causal(argmind::Policy::Interpolate));
}

void test_drop_if_stale_is_the_only_policy_that_refuses() {
    const argmind::Aligner hold(argmind::Policy::HoldLast, 0.5);
    const argmind::Aligner drop(argmind::Policy::DropIfStale, 0.5);
    const argmind::Aligner unlimited(argmind::Policy::DropIfStale, 0.0);
    CHECK(!hold.refuses_stale(-10.0));
    CHECK(drop.refuses_stale(-10.0));
    CHECK(drop.refuses_stale(10.0));       // the magnitude is what matters
    CHECK(!drop.refuses_stale(-0.25));
    CHECK(!unlimited.refuses_stale(-1e9));  // a non-positive limit means no limit
}

void test_the_declared_ring_memory_matches_the_shape() {
    using S = argmind::Stream<float, 6, 64>;
    CHECK(S::payload_bytes == 64 * 6 * 4 + 64 * 8 + 64);
    CHECK(argmind::Stream<double, 1, 8>::payload_bytes == 8 * 1 * 8 + 8 * 8 + 8);

    // The constant restated its own definition and never met `sizeof`, while the header
    // told a caller to budget from it — 24 bytes short per stream on this host. What is
    // true and worth pinning is the relation: the payload is part of the object, never the
    // whole of it, and the gap is bookkeeping plus padding rather than anything unbounded.
    CHECK(sizeof(S) >= static_cast<size_t>(S::payload_bytes));
    CHECK(sizeof(S) - static_cast<size_t>(S::payload_bytes) < 64u);
    CHECK(sizeof(argmind::Stream<double, 1, 8>) >=
          static_cast<size_t>(argmind::Stream<double, 1, 8>::payload_bytes));
}


// -- pipeline.hpp -------------------------------------------------------------
//
// `cpp/tools/pipecheck.cpp` drives whole runs and compares them against the reference, which
// is the right shape for bookkeeping under interleaving and reaches almost none of the
// *refusals*. A configuration the pipeline rejects produces no run to compare, so every
// `configure()` guard, the frame-map arithmetic and the two prediction paths are invisible to
// it. That gap is not hypothetical: the last port shipped with `Stream<>::push` covered only
// by a copy of itself, and a parity harness can only test what its tool exercises.

void test_a_frame_map_must_fill_the_adapter_exactly() {
    argmind::FrameMap<3, 1> map;
    CHECK(!map.valid());                  // nothing declared
    CHECK(map.add_feature(1, 0));
    CHECK(map.add_feature(1, 1));
    CHECK(map.add_target(0, 0));
    CHECK(!map.valid());                  // two features, three wanted
    map.set_constant(true);
    CHECK(map.valid());                   // the constant makes up the difference

    // A short map would leave an element of x holding whatever the previous frame put there,
    // which is a wrong answer rather than a missing feature.
    argmind::FrameMap<3, 1> over;
    CHECK(over.add_feature(0, 0));
    CHECK(over.add_feature(0, 1));
    CHECK(over.add_feature(0, 2));
    CHECK(!over.add_feature(0, 3));       // full; refused rather than written past the end
    CHECK(over.add_target(0, 0));
    CHECK(!over.add_target(0, 1));        // M is 1
    CHECK(over.valid());
    over.set_constant(true);
    CHECK(!over.valid());                 // now four features for a three-feature adapter

    CHECK(!over.add_feature(-1, 0));
    CHECK(!over.add_feature(0, -1));
}

namespace pipe_fixture {

using Rls6 = argmind::Rls<float, 3, 1>;
using Pipe = argmind::Pipeline<Rls6, 3>;

/// Two streams: s0 the 1-wide reference carrying the target, s1 a 2-wide sensor.
struct Rig {
    argmind::Stream<float, 1, 8> ref;
    argmind::Stream<float, 2, 8> aux;
    argmind::StreamView<float> views[2];

    void refresh() {
        views[0] = ref.view();
        views[1] = aux.view();
    }
};

argmind::FrameMap<3, 1> good_map() {
    argmind::FrameMap<3, 1> map;
    map.add_feature(1, 0);
    map.add_feature(1, 1);
    map.add_target(0, 0);
    map.set_constant(true);
    return map;
}

Pipe make(argmind::Policy policy = argmind::Policy::HoldLast) {
    return Pipe(Rls6(0.99f, 10.0f), argmind::Aligner(policy, 0.0), good_map(), 0);
}

}  // namespace pipe_fixture

void test_configure_refuses_what_the_reference_raises_on() {
    using namespace pipe_fixture;
    Rig rig;
    rig.refresh();

    // A reference index outside the stream set.
    {
        Pipe p(Rls6(0.99f, 10.0f), argmind::Aligner(argmind::Policy::HoldLast, 0.0),
               good_map(), 2);
        CHECK(!p.configure(rig.views, 2));
        CHECK(!p.configured());
    }
    // A component past the end of the sensor it names. The reference raises at construction
    // for the same reason: at 3am on the first frame is the wrong time to find out.
    {
        argmind::FrameMap<3, 1> map;
        map.add_feature(1, 0);
        map.add_feature(1, 5);            // s1 is 2 wide
        map.add_target(0, 0);
        map.set_constant(true);
        Pipe p(Rls6(0.99f, 10.0f), argmind::Aligner(argmind::Policy::HoldLast, 0.0), map, 0);
        CHECK(!p.configure(rig.views, 2));
    }
    // A non-causal policy in a causal loop. Legitimate offline, a correctness bug here, so it
    // has to be said out loud even though the aligner alone permits it.
    {
        Pipe p = make(argmind::Policy::Interpolate);
        CHECK(!p.configure(rig.views, 2));
        p.set_allow_non_causal(true);
        CHECK(p.configure(rig.views, 2));
    }
    // A zero staleness limit. `Aligner` folds "no limit" and "a limit of zero" into one
    // non-positive sentinel and that is recorded there as a real difference from the
    // reference; this gate deliberately does not repeat it, so a caller meaning
    // zero-tolerance is refused rather than silently given no check at all.
    {
        Pipe p = make();
        p.set_max_stale_s(0.0);
        CHECK(!p.configure(rig.views, 2));
        p.set_max_stale_s(0.5);
        CHECK(p.configure(rig.views, 2));
        p.clear_max_stale_s();
        CHECK(p.configure(rig.views, 2));
    }
    {
        Pipe p = make();
        CHECK(!p.configure(nullptr, 2));
        CHECK(!p.configure(rig.views, 0));
        CHECK(!p.configure(rig.views, 4));   // more than MaxStreams
    }
}

void test_an_unconfigured_step_reports_that_rather_than_idling() {
    using namespace pipe_fixture;
    Rig rig;
    rig.refresh();
    Pipe p = make();
    const auto rep = p.step(rig.views);
    // `ok == false` and `idle()` must not be confused: one means nothing arrived, the other
    // means nothing was attempted. A pipeline that never ran would otherwise look healthy
    // forever.
    CHECK(!rep.ok);
    CHECK(!rep.healthy());
    CHECK(rep.idle());
    CHECK(p.steps() == 0);
}

void test_a_view_whose_width_changed_is_refused() {
    using namespace pipe_fixture;
    Rig rig;
    rig.refresh();
    Pipe p = make();
    CHECK(p.configure(rig.views, 2));

    const float v[2] = {1.0f, 2.0f};
    const float y[1] = {3.0f};
    CHECK(rig.aux.push(0.5, v));
    CHECK(rig.ref.push(1.0, y));
    rig.refresh();
    CHECK(p.step(rig.views).frames == 1);

    // A different stream wearing the same index. The map's components were bounds-checked
    // against the old width, so proceeding would read past the end of the ring.
    argmind::StreamView<float> lied[2] = {rig.views[0], rig.views[1]};
    lied[1].dim = 1;
    const auto rep = p.step(lied);
    CHECK(!rep.ok);
    CHECK(rep.frames == 0);
}

void test_a_partial_frame_is_not_predictable_from() {
    using namespace pipe_fixture;
    Rig rig;
    rig.refresh();
    Pipe p = make();
    CHECK(p.configure(rig.views, 2));
    CHECK(!p.frame_ready());

    const float v[2] = {1.0f, 2.0f};
    const float y[1] = {3.0f};
    // s0 gets an early sample s1 cannot answer for, which is what makes the failure PARTIAL
    // rather than immediate: the reference is stream 0 and is extracted first, so a frame
    // that dies on stream 1 has already written y.
    CHECK(rig.ref.push(0.2, y));
    CHECK(rig.aux.push(0.5, v));
    CHECK(rig.ref.push(1.0, y));
    rig.refresh();
    CHECK(p.step(rig.views).frames == 1);
    CHECK(p.frame_ready());
    float out[1] = {0.0f};
    CHECK(p.predict_last(out));

    // t = 0.3: s0 answers with its 0.2 sample and s1 has nothing yet. y is now this
    // instant's and x is still the last frame's -- exactly the mixture that predicts a
    // plausible wrong answer with no error anywhere.
    CHECK(!p.predict_at(rig.views, 0.3, out));
    CHECK(!p.frame_ready());
    CHECK(!p.predict_last(out));
}

void test_predicting_consumes_nothing() {
    using namespace pipe_fixture;
    Rig rig;
    rig.refresh();
    Pipe p = make();
    CHECK(p.configure(rig.views, 2));

    const float v[2] = {1.0f, 2.0f};
    const float y[1] = {3.0f};
    CHECK(rig.aux.push(0.5, v));
    CHECK(rig.ref.push(1.0, y));
    rig.refresh();
    CHECK(p.step(rig.views).frames == 1);

    const uint64_t frames = p.aligner().frames();
    const uint64_t no_data = p.aligner().no_data();
    const double watermark = p.aligner().consumed_to();
    float out[1] = {0.0f};
    CHECK(p.predict_at(rig.views, 1.0, out));
    CHECK(!p.predict_at(rig.views, 0.25, out));   // before s1 has anything
    // A prediction is not a consumption. If it moved the watermark it would eat a frame the
    // next step is owed; if it moved `no_data` the pipeline's own loss arithmetic would drift.
    CHECK(p.aligner().frames() == frames);
    CHECK(p.aligner().no_data() == no_data);
    CHECK(p.aligner().consumed_to() == watermark);
}

void test_the_guarded_and_unguarded_forms_are_both_constructible() {
    using namespace pipe_fixture;
    // The Python decides this with getattr; here it is a trait, and the reason is the same:
    // the reference has to be runnable UNGUARDED to serve as the oracle this core is verified
    // against. Guarding is the right default in the field and is not enforced in the type.
    using Guarded = argmind::Guard<Rls6, 2, 4>;
    CHECK(!argmind::Pipeline<Rls6, 3>::guarded);
    CHECK(argmind::Pipeline<Guarded, 3>::guarded);

    Guarded g(Rls6(0.99f, 10.0f), 1, 0);
    CHECK(g.add_bound(argmind::Bound::at_most(argmind::Field::PTrace, 1e6)));
    CHECK(g.valid());
    argmind::Pipeline<Guarded, 3> p(g, argmind::Aligner(argmind::Policy::HoldLast, 0.0),
                                    good_map(), 0);
    Rig rig;
    rig.refresh();
    CHECK(p.configure(rig.views, 2));
    CHECK(!p.frozen());
}

void test_a_step_with_nothing_to_visit_touches_no_counter() {
    using namespace pipe_fixture;
    // What this pins: a step that finds no unconsumed reference instant increments nothing.
    // Not the loss arithmetic -- that is pinned in tests/test_cpp_pipeline_parity.py, by the
    // one case where declined instants and an overflowing ring appear in the same run.
    // Separating the two matters because both were nearly tested here by accident and
    // neither would have been.
    //
    // And a finding recorded where it lives rather than tallied. `lost = arrived -
    // considered` is clamped at zero on both sides, and the clamp is DEAD: each visited
    // instant increments exactly one of frames/no_data/dropped_stale, so `considered` is the
    // count of instants visited, and every unconsumed reference sample in the ring arrived
    // after the last successful step -- `considered <= owed` holds by construction. A
    // mutation check confirms it: removing the clamp survives the whole suite, including a
    // test written specifically to kill it, and including advancing the aligner from outside
    // between steps (which cannot reach it either, since the deltas are sampled inside the
    // step). It stays because an unsigned subtraction that wrapped would report about 1.8e19
    // lost frames, and a branch nobody can reach is a cheap price for that.
    Rig rig;
    rig.refresh();
    Pipe p = make();
    CHECK(p.configure(rig.views, 2));

    const float v[2] = {1.0f, 2.0f};
    const float y[1] = {3.0f};
    CHECK(rig.aux.push(0.5, v));
    CHECK(rig.ref.push(1.0, y));
    rig.refresh();
    CHECK(p.step(rig.views).frames == 1);

    p.aligner_mutable().note_no_data();
    p.aligner_mutable().note_dropped_stale();
    const auto rep = p.step(rig.views);   // the only instant there is has been consumed
    CHECK(rep.ok);
    CHECK(rep.frames == 0);
    CHECK(rep.lost_frames == 0);
    CHECK(p.aligner().frames() == 1);
    CHECK(p.aligner().no_data() == 1);        // exactly the one injected above
    CHECK(p.aligner().dropped_stale() == 1);  // likewise
}

void test_the_declared_pipeline_memory_is_what_it_says() {
    using namespace pipe_fixture;
    // The blend buffer is the one allocation-shaped thing here and it is larger than it
    // looks: `Pick<T>` carries MAX_DIM elements whether or not the policy interpolates. One
    // of these is held, not MaxStreams of them -- that is what fusing the frame loop bought,
    // and the number is here so a change to it shows up as a failing test rather than as
    // quietly more RAM.
    CHECK(Pipe::pick_bytes == static_cast<int>(sizeof(argmind::Pick<float>)));
    CHECK(sizeof(Pipe) >= static_cast<size_t>(Pipe::pick_bytes));
    CHECK(sizeof(Pipe) < static_cast<size_t>(Pipe::pick_bytes) + 512u);
}


// -- the guard and record ABI -------------------------------------------------
//
// What these test is the PLUMBING, not the algorithm: guard.hpp and persist.hpp are
// already verified against the Python reference, so the question at the boundary is
// whether the shim carries the same answers across it. The strongest form of that
// question is to run the same inputs through both paths and demand agreement bit for
// bit, which is what the first test does -- a tolerance here would be measuring
// nothing, since it is the same code on both sides of the comparison.

void test_the_guard_abi_agrees_with_the_header_bit_for_bit() {
    static am_rls_storage store[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_guard_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 2, 0) == AM_OK);
    CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_RESIDUAL_NORM, 0, 0.0, 1, 6.0) ==
          AM_OK);
    CHECK(am_guard_f32_6_1_set_divergence(store, 4.0, 0.4, 0.05, 1e-3, 10) == AM_OK);
    CHECK(am_guard_f32_6_1_valid(store) == AM_OK);

    argmind::Guard<argmind::Rls<float, 6, 1>, 4, 4> ref(
        argmind::Rls<float, 6, 1>(0.98f, 10.0f), 2, 0);
    CHECK(ref.add_bound(argmind::Bound::at_most(argmind::Field::ResidualNorm, 6.0)));
    argmind::Divergence d;
    d.ratio = 4.0;
    d.fast_alpha = 0.4;
    d.slow_alpha = 0.05;
    d.floor = 1e-3;
    d.warmup = 10;
    ref.set_divergence(d);
    CHECK(ref.valid());

    uint32_t seed = 12345u;
    for (int k = 0; k < 120; ++k) {
        float x[6];
        for (int j = 0; j < 6; ++j) {
            seed = seed * 1664525u + 1013904223u;
            x[j] = static_cast<float>(static_cast<int32_t>(seed >> 8) % 2000) / 1000.0f;
        }
        // A fault at step 60, so the trip, the rollback and the frozen probes all cross
        // the boundary rather than only the healthy path.
        float y[1] = {2.0f * x[0] - x[3] + 0.5f + (k == 60 ? 40.0f : 0.0f)};

        am_guard_report r;
        float residual[1];
        CHECK(am_guard_f32_6_1_update(store, x, y, &r, residual) == AM_OK);
        const argmind::GuardReport<float, 1> want = ref.update(x, y);

        CHECK(r.applied == (want.applied ? 1 : 0));
        CHECK(r.reason == static_cast<int32_t>(want.reason));
        CHECK(r.frozen == (want.frozen ? 1 : 0));
        CHECK(r.rolled_back == (want.rolled_back ? 1 : 0));
        CHECK(r.tripped == want.tripped);
        CHECK(r.guard_reason == static_cast<int32_t>(want.guard_reason));
        CHECK(bits_equal(r.residual_norm, want.residual_norm));
        CHECK(bits_equal(r.p_trace, want.p_trace));
        CHECK(bits_equal(r.gain_norm, want.gain_norm));
        CHECK(residual[0] == want.residual[0]);
    }
    // The fixture has to have exercised the fault, or every CHECK above ran on the
    // healthy path and the frozen and rolled-back fields were compared as false == false.
    CHECK(am_guard_f32_6_1_frozen(store) == 1);
    CHECK(am_guard_f32_6_1_frozen_by(store) != 0u);

    am_guard_counters c;
    CHECK(am_guard_f32_6_1_counters(store, &c) == AM_OK);
    CHECK(c.trips == 1u);
    CHECK(c.rollbacks == 1u);
    CHECK(c.blocked_frozen > 0u);
    CHECK(c.applied + c.trips + c.blocked_frozen + c.rejected_by_adapter == 120u);

    const float* a = am_guard_f32_6_1_state(store);
    const float* b = ref.state();
    for (int i = 0; i < am_guard_f32_6_1_state_size(); ++i) CHECK(a[i] == b[i]);
}

void test_the_guard_abi_refuses_at_the_boundary_rather_than_trusting() {
    static am_rls_storage store[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_guard_f32_6_1_init(nullptr, 64, 0.98f, 10.0f, 1, 0) == AM_ERR_NULL);
    CHECK(am_guard_f32_6_1_init(store, 8, 0.98f, 10.0f, 1, 0) == AM_ERR_TOO_SMALL);
    CHECK(am_guard_f32_6_1_init(store, sizeof(store), 0.0f, 10.0f, 1, 0) ==
          AM_ERR_CONFIG);
    CHECK(am_guard_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 1, 0) == AM_OK);

    // A guard with no checks can never freeze, which makes it a wrapper that looks like a
    // safety mechanism. Refused, as the header path refuses it.
    CHECK(am_guard_f32_6_1_valid(store) == AM_ERR_UNUSABLE);

    // A field code the enum does not have. The C++ side cannot make this mistake because
    // the enum is the parameter; the C side can, and casting it in would produce a check
    // that watches nothing and never fires.
    CHECK(am_guard_f32_6_1_add_bound(store, 99, 0, 0.0, 1, 1.0) == AM_ERR_CONFIG);
    CHECK(am_guard_f32_6_1_add_bound(store, -1, 0, 0.0, 1, 1.0) == AM_ERR_CONFIG);
    // A bound that sets neither end can never fail.
    CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_P_TRACE, 0, 0.0, 0, 0.0) ==
          AM_ERR_CONFIG);
    // lo above hi.
    CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_P_TRACE, 1, 9.0, 1, 1.0) ==
          AM_ERR_CONFIG);
    // A divergence whose "fast" average is the slower of the two detects the opposite of
    // what it claims to.
    CHECK(am_guard_f32_6_1_set_divergence(store, 4.0, 0.02, 0.4, 0.0, 10) ==
          AM_ERR_CONFIG);
    CHECK(am_guard_f32_6_1_set_divergence(store, 0.5, 0.4, 0.02, 0.0, 10) ==
          AM_ERR_CONFIG);

    for (int i = 0; i < 4; ++i) {
        CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_P_TRACE, 0, 0.0, 1,
                                         1e9 + i) == AM_OK);
    }
    CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e9) ==
          AM_ERR_FULL);
    CHECK(am_guard_f32_6_1_valid(store) == AM_OK);

    float x[6] = {1, 0, 0, 0, 0, 0};
    float y[1] = {1};
    am_guard_report r;
    CHECK(am_guard_f32_6_1_update(nullptr, x, y, &r, nullptr) == AM_ERR_NULL);
    CHECK(am_guard_f32_6_1_update(store, x, y, nullptr, nullptr) == AM_ERR_NULL);
    CHECK(am_guard_f32_6_1_state(nullptr) == nullptr);
    CHECK(am_guard_f32_6_1_counters(store, nullptr) == AM_ERR_NULL);
    CHECK(am_guard_f32_6_1_frozen(nullptr) == 0);
}

void test_the_declared_guard_bounds_are_bounds_and_not_fantasies() {
    // A bound that is merely true is not useful: `sizeof <= 99999` would pass forever
    // while telling a caller to reserve 100 KB. The second half of each pair is what
    // keeps the promise tight enough to budget from.
    CHECK(am_guard_f32_6_1_bytes() <= AM_GUARD_F32_6_1_MAX_BYTES);
    CHECK(am_guard_f32_6_3_bytes() <= AM_GUARD_F32_6_3_MAX_BYTES);
    CHECK(am_guard_f32_12_1_bytes() <= AM_GUARD_F32_12_1_MAX_BYTES);
    CHECK(am_guard_f64_6_1_bytes() <= AM_GUARD_F64_6_1_MAX_BYTES);
    CHECK(am_guard_f32_6_1_bytes() * 2 >= AM_GUARD_F32_6_1_MAX_BYTES);
    CHECK(am_guard_f32_12_1_bytes() * 2 >= AM_GUARD_F32_12_1_MAX_BYTES);

    // Where the memory goes, reported rather than inferred. The ring is the majority of
    // it at every shape, which is the thing a caller weighing rollback depth needs to see.
    CHECK(am_guard_f32_6_1_checkpoint_bytes() ==
          4 * (AM_RLS_F32_6_1_STATE * 4 + 16));
    CHECK(am_guard_f32_6_1_checkpoint_bytes() * 2 >
          static_cast<int>(am_guard_f32_6_1_bytes()));
    CHECK(am_guard_f32_6_1_state_size() == AM_RLS_F32_6_1_STATE);
    CHECK(am_guard_f64_6_1_state_size() == AM_RLS_F64_6_1_STATE);
}

void test_the_record_abi_round_trips_and_arbitrates() {
    static float state[AM_RLS_F32_6_1_STATE];
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i) state[i] = 0.25f * i - 3.0f;
    const char cause[] = "p_trace";
    const size_t need = am_persist_record_bytes(AM_RLS_F32_6_1_STATE, sizeof(float),
                                                sizeof(cause) - 1);
    static uint8_t slot0[512];
    static uint8_t slot1[512];
    CHECK(need <= sizeof(slot0));

    CHECK(am_persist_pack_f32(state, AM_RLS_F32_6_1_STATE, 6, 1, 7, 0xABCD1234u, 1,
                              cause, sizeof(cause) - 1, slot0, need) == need);
    // One byte short must write nothing rather than a truncated record: a slot that is
    // half a record is the torn write two slots exist to survive.
    CHECK(am_persist_pack_f32(state, AM_RLS_F32_6_1_STATE, 6, 1, 7, 0xABCD1234u, 1,
                              cause, sizeof(cause) - 1, slot1, need - 1) == 0u);

    am_record rec;
    CHECK(am_persist_unpack_f32(slot0, need, &rec) == AM_PERSIST_OK);
    CHECK(rec.sequence == 7u);
    CHECK(rec.fingerprint == 0xABCD1234u);
    CHECK(rec.n_features == 6 && rec.n_outputs == 1);
    CHECK(rec.frozen == 1);
    CHECK(rec.count == static_cast<uint32_t>(AM_RLS_F32_6_1_STATE));
    CHECK(rec.cause_len == sizeof(cause) - 1);
    CHECK(rec.cause != nullptr && rec.cause[0] == 'p');

    static float back[AM_RLS_F32_6_1_STATE];
    CHECK(am_persist_read_state_f32(slot0, need, back, AM_RLS_F32_6_1_STATE) ==
          AM_PERSIST_OK);
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i) CHECK(back[i] == state[i]);
    CHECK(am_persist_read_state_f32(slot0, need, back, AM_RLS_F32_6_1_STATE - 1) ==
          AM_PERSIST_LENGTH_MISMATCH);

    // A single flipped bit anywhere must be refused, which is the whole reason the CRC is
    // there. Checked at three places rather than one, because the header, the payload and
    // the CRC field itself are protected by different parts of the check.
    const size_t spots[3] = {6, need / 2, need - 1};
    for (int i = 0; i < 3; ++i) {
        slot0[spots[i]] ^= 0x01u;
        CHECK(am_persist_unpack_f32(slot0, need, &rec) != AM_PERSIST_OK);
        slot0[spots[i]] ^= 0x01u;
    }
    CHECK(am_persist_unpack_f32(slot0, need, &rec) == AM_PERSIST_OK);

    // Arbitration, and the rule the reference got wrong: the next write goes to the slot
    // that is NOT newest, or a reboot after an odd number of saves destroys the record it
    // has just loaded.
    const size_t n1 = am_persist_pack_f32(state, AM_RLS_F32_6_1_STATE, 6, 1, 9,
                                          0xABCD1234u, 0, nullptr, 0, slot1,
                                          sizeof(slot1));
    CHECK(n1 > 0u);
    CHECK(n1 < need);          // no cause this time, so a shorter record
    int32_t sa = -1;
    int32_t sb = -1;
    // `in_bytes` is the RECORD's length, not the buffer's capacity -- `unpack` compares it
    // exactly, so a slot read into an oversized buffer is rejected as a length mismatch.
    // That is the right refusal (a torn write looks exactly like a short record) and it is
    // a real constraint on how a caller sizes flash slots, now said in argmind.h.
    CHECK(am_persist_unpack_f32(slot1, sizeof(slot1), &rec) ==
          AM_PERSIST_LENGTH_MISMATCH);
    const int newest = am_persist_select_newest_f32(slot0, need, slot1, n1,
                                                    &rec, &sa, &sb);
    CHECK(newest == 1);
    CHECK(rec.sequence == 9u);
    CHECK(sa == AM_PERSIST_OK && sb == AM_PERSIST_OK);
    CHECK(am_persist_next_slot(newest) == 0);
    CHECK(am_persist_next_slot(-1) == 0);

    // Both slots unreadable is a first boot, not an error to abort on -- and WHY each was
    // rejected comes back, because "it started from scratch" and "it started from scratch
    // because slot 1 had a CRC mismatch" are very different mornings.
    static uint8_t junk[64];
    for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = 0xFFu;
    CHECK(am_persist_select_newest_f32(junk, sizeof(junk), nullptr, 0, &rec, &sa, &sb) ==
          -1);
    CHECK(sa == AM_PERSIST_BAD_MAGIC);
    CHECK(sb == AM_PERSIST_TOO_SHORT);

    // A record packed as f32 must not be read as f64. Nothing downstream could detect it:
    // the bytes are a valid block of half the length.
    CHECK(am_persist_unpack_f64(slot0, need, &rec) == AM_PERSIST_BAD_DTYPE);
}

void test_the_abi_closes_the_loop_from_a_guard_to_a_slot_and_back() {
    // The reason this section exists at all: `am_rls_*_state()` handed out the block a
    // device must survive a power cut with, and there was no way to frame it from C. This
    // is that loop end to end -- adapt, pack, reboot, arbitrate, restore -- because each
    // half working says nothing about the two meeting.
    static am_rls_storage store[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_guard_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 1, 0) == AM_OK);
    CHECK(am_guard_f32_6_1_add_bound(store, AM_FIELD_P_TRACE, 0, 0.0, 1, 1e6) == AM_OK);

    uint32_t seed = 99u;
    for (int k = 0; k < 40; ++k) {
        float x[6];
        for (int j = 0; j < 6; ++j) {
            seed = seed * 1664525u + 1013904223u;
            x[j] = static_cast<float>(static_cast<int32_t>(seed >> 8) % 1000) / 500.0f;
        }
        const float y[1] = {x[0] + 2.0f * x[1] - 0.5f};
        am_guard_report r;
        CHECK(am_guard_f32_6_1_update(store, x, y, &r, nullptr) == AM_OK);
    }
    static float saved[AM_RLS_F32_6_1_STATE];
    const float* live = am_guard_f32_6_1_state(store);
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i) saved[i] = live[i];

    static uint8_t slot[512];
    const size_t n = am_persist_pack_f32(saved, AM_RLS_F32_6_1_STATE, 6, 1, 1,
                                         0xFEEDu, am_guard_f32_6_1_frozen(store),
                                         nullptr, 0, slot, sizeof(slot));
    CHECK(n > 0u);

    // Reboot: a fresh guard, restored from the slot.
    static am_rls_storage fresh[AM_GUARD_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_guard_f32_6_1_init(fresh, sizeof(fresh), 0.98f, 10.0f, 1, 0) == AM_OK);
    static float restored[AM_RLS_F32_6_1_STATE];
    CHECK(am_persist_read_state_f32(slot, n, restored, AM_RLS_F32_6_1_STATE) ==
          AM_PERSIST_OK);
    CHECK(am_guard_f32_6_1_load_state(fresh, restored) == AM_OK);

    const float* a = am_guard_f32_6_1_state(store);
    const float* b = am_guard_f32_6_1_state(fresh);
    for (int i = 0; i < AM_RLS_F32_6_1_STATE; ++i) CHECK(a[i] == b[i]);

    // ...and the same input now predicts the same thing on both, which is the property a
    // caller actually cares about and is not implied by the blocks matching only if the
    // restore silently reset something else.
    const float probe[6] = {0.3f, -1.2f, 0.75f, 0.0f, 2.5f, -0.4f};
    float pa[1];
    float pb[1];
    CHECK(am_guard_f32_6_1_predict(store, probe, pa) == AM_OK);
    CHECK(am_guard_f32_6_1_predict(fresh, probe, pb) == AM_OK);
    CHECK(pa[0] == pb[0]);

    // A non-finite block must be refused rather than poisoning every later update: flash
    // is exactly where a torn write or a bit flip shows up.
    restored[3] = __builtin_nanf("");
    CHECK(am_guard_f32_6_1_load_state(fresh, restored) == AM_ERR_STATE);
}


// -- the stream and pipeline ABI ----------------------------------------------
//
// The composed handle is the piece the previous ABI section said was missing, and the way it
// avoids the instantiation explosion is that the caller owns the sample storage: `dim` and
// `capacity` are runtime values on the C side. That makes one question worth asking hardest --
// does the runtime-shaped ring behave exactly like `Stream<T, Dim, Capacity>`? It should by
// construction, since both call `ring_push`, but "by construction" is what this project keeps
// finding to be untrue.

void test_the_caller_owned_ring_behaves_exactly_like_the_typed_one() {
    static float values[8 * 2];
    static double times[8];
    static uint8_t valid[8];
    am_ring_f32 r;
    CHECK(am_ring_f32_init(&r, values, times, valid, 2, 8) == AM_OK);
    argmind::Stream<float, 2, 8> s;

    const double stamps[10] = {1.0, 1.0, 0.5, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    for (int k = 0; k < 10; ++k) {
        const float v[2] = {static_cast<float>(k), -static_cast<float>(k)};
        const bool ok = (k % 3) != 0;
        const int a = am_ring_f32_push(&r, stamps[k], v, ok ? 1 : 0);
        const bool b = s.push(stamps[k], v, ok);
        // Refusals have to match too: from the first disagreement the two hold different
        // histories and every later comparison is against the wrong thing.
        CHECK((a == AM_OK) == b);
    }
    // ...and the non-finite cases, which is where the Python reference had a real defect.
    const float v[2] = {0.0f, 0.0f};
    CHECK((am_ring_f32_push(&r, __builtin_nanf(""), v, 1) == AM_OK) ==
          s.push(__builtin_nanf(""), v));
    CHECK((am_ring_f32_push(&r, __builtin_inff(), v, 1) == AM_OK) ==
          s.push(__builtin_inff(), v));

    CHECK(r.count == s.size());
    CHECK(r.dropped == s.dropped());
    const argmind::StreamView<float> ra =
        argmind::ring_view<float>(*reinterpret_cast<argmind::Ring<float>*>(&r));
    const argmind::StreamView<float> rb = s.view();
    CHECK(ra.head == rb.head);
    for (int age = 0; age < ra.count; ++age) {
        CHECK(ra.at(age).t == rb.at(age).t);
        CHECK(ra.at(age).valid == rb.at(age).valid);
        CHECK(ra.at(age).values[0] == rb.at(age).values[0]);
        CHECK(ra.at(age).values[1] == rb.at(age).values[1]);
    }

    // A freshly initialised ring must accept t = 0 and negative times, because `last_t`
    // starts at -inf and not at zero. Found by a mutation check: setting it to 0.0 survived
    // the whole suite, and it would silently refuse the first sample of any device whose
    // timebase is milliseconds-since-boot -- which is most of them. `Stream<>::reset` had
    // this right; nothing compared the two on a fresh ring until now.
    am_ring_f32 zero;
    static float zv[4 * 1];
    static double zt[4];
    static uint8_t zk[4];
    argmind::Stream<float, 1, 4> zs;
    CHECK(am_ring_f32_init(&zero, zv, zt, zk, 1, 4) == AM_OK);
    const float one[1] = {1.0f};
    CHECK((am_ring_f32_push(&zero, -5.0, one, 1) == AM_OK) == zs.push(-5.0, one));
    CHECK((am_ring_f32_push(&zero, 0.0, one, 1) == AM_OK) == zs.push(0.0, one));
    CHECK(zero.count == 2 && zs.size() == 2);
    CHECK(am_ring_f32_reset(&zero) == AM_OK);
    zs.reset();
    CHECK((am_ring_f32_push(&zero, 0.0, one, 1) == AM_OK) == zs.push(0.0, one));
    CHECK(zero.count == 1 && zs.size() == 1);

    CHECK(am_ring_f32_init(&r, nullptr, times, valid, 2, 8) == AM_ERR_NULL);
    CHECK(am_ring_f32_init(&r, values, times, valid, 0, 8) == AM_ERR_CONFIG);
    CHECK(am_ring_f32_init(&r, values, times, valid, 2, 0) == AM_ERR_CONFIG);
    CHECK(am_ring_f32_push(nullptr, 1.0, v, 1) == AM_ERR_NULL);
}

void test_the_pipeline_abi_agrees_with_the_header_bit_for_bit() {
    // Three streams of different widths and rates: s0 the 1-wide reference carrying the
    // target, s1 a 3-wide sensor and s2 a 2-wide one, both phase-offset so a reference
    // instant never lands on one of their samples. A single-rate fixture would agree under
    // any alignment policy and prove nothing about the wiring.
    //
    // The target is NOT among the features, which the first version of this fixture got
    // wrong: it mapped (0, 0) -- the reference stream's own value -- into the regressor, so
    // the model read the answer off its own input, the residual sat at zero, and the fault
    // injected to trip the guard did nothing. The tell was three failing CHECKs at the end
    // saying nothing had frozen, and everything else agreeing perfectly. A fixture that
    // agrees while testing nothing is the shape this project keeps finding.
    static float v0[16 * 1];
    static double t0[16];
    static uint8_t k0[16];
    static float v1[16 * 3];
    static double t1[16];
    static uint8_t k1[16];
    static float v2[16 * 2];
    static double t2[16];
    static uint8_t k2[16];
    am_ring_f32 rings[3];
    CHECK(am_ring_f32_init(&rings[0], v0, t0, k0, 1, 16) == AM_OK);
    CHECK(am_ring_f32_init(&rings[1], v1, t1, k1, 3, 16) == AM_OK);
    CHECK(am_ring_f32_init(&rings[2], v2, t2, k2, 2, 16) == AM_OK);

    static am_rls_storage store[AM_PIPE_G_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_pipe_g_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 2, 0,
                                 AM_POLICY_HOLD_LAST, 0.0, 0) == AM_OK);
    // Interleaved across the two sensors on purpose: the C++ extracts by scattering from one
    // stream at a time, so a map that took every component from one stream would agree under
    // a transposed scatter too.
    CHECK(am_pipe_g_f32_6_1_add_feature(store, 1, 0) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_feature(store, 2, 1) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_feature(store, 1, 2) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_feature(store, 2, 0) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_feature(store, 1, 1) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_target(store, 0, 0) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_set_constant(store, 1) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_add_bound(store, AM_FIELD_RESIDUAL_NORM, 0, 0.0, 1, 8.0) ==
          AM_OK);
    CHECK(am_pipe_g_f32_6_1_set_max_stale_s(store, 1, 2.0) == AM_OK);
    CHECK(am_pipe_g_f32_6_1_configure(store, rings, 3) == AM_OK);

    // The same wiring through the header.
    using Adapter = argmind::Guard<argmind::Rls<float, 6, 1>, 4, 4>;
    argmind::Stream<float, 1, 16> s0;
    argmind::Stream<float, 3, 16> s1;
    argmind::Stream<float, 2, 16> s2;
    argmind::FrameMap<6, 1> map;
    map.add_feature(1, 0);
    map.add_feature(2, 1);
    map.add_feature(1, 2);
    map.add_feature(2, 0);
    map.add_feature(1, 1);
    map.add_target(0, 0);
    map.set_constant(true);
    Adapter g(argmind::Rls<float, 6, 1>(0.98f, 10.0f), 2, 0);
    CHECK(g.add_bound(argmind::Bound::at_most(argmind::Field::ResidualNorm, 8.0)));
    argmind::Pipeline<Adapter, 4> ref(
        g, argmind::Aligner(argmind::Policy::HoldLast, 0.0), map, 0);
    ref.set_max_stale_s(2.0);
    argmind::StreamView<float> views[3] = {s0.view(), s1.view(), s2.view()};
    CHECK(ref.configure(views, 3));

    uint32_t seed = 4242u;
    int steps = 0;
    int with_frames = 0;
    for (int k = 1; k <= 60; ++k) {
        float a[3];
        float b[2];
        for (int j = 0; j < 3; ++j) {
            seed = seed * 1664525u + 1013904223u;
            a[j] = static_cast<float>(static_cast<int32_t>(seed >> 9) % 1000) / 500.0f;
        }
        for (int j = 0; j < 2; ++j) {
            seed = seed * 1664525u + 1013904223u;
            b[j] = static_cast<float>(static_cast<int32_t>(seed >> 9) % 1000) / 500.0f;
        }
        // Occasionally invalid rather than routinely: `require_valid` is on, and marking the
        // newest sample before each reference instant bad made EVERY frame skip the learner.
        const bool ok = (k % 11) != 0;
        CHECK((am_ring_f32_push(&rings[1], k - 0.85, a, ok ? 1 : 0) == AM_OK) ==
              s1.push(k - 0.85, a, ok));
        CHECK((am_ring_f32_push(&rings[2], k - 0.45, b, 1) == AM_OK) ==
              s2.push(k - 0.45, b, true));

        // A model the adapter can actually learn, so the ordinary residual settles well
        // below the bound and the injected fault is the only thing that crosses it.
        const float y[1] = {2.0f * a[0] - b[1] + 0.5f * a[2] + 0.25f * b[0] - 1.5f * a[1] +
                            0.3f + (k == 35 ? 90.0f : 0.0f)};
        CHECK((am_ring_f32_push(&rings[0], double(k), y, 1) == AM_OK) ==
              s0.push(double(k), y, true));

        if (k % 3 != 0) continue;
        am_step_report got;
        CHECK(am_pipe_g_f32_6_1_step(store, rings, 3, &got) == AM_OK);
        views[0] = s0.view();
        views[1] = s1.view();
        views[2] = s2.view();
        const argmind::StepReport<4> want = ref.step(views);
        ++steps;
        if (want.frames) ++with_frames;

        CHECK(got.ok == (want.ok ? 1 : 0));
        CHECK(got.frames == want.frames);
        CHECK(got.updated == want.updated);
        CHECK(got.lost_frames == want.lost_frames);
        CHECK(got.frozen == (want.frozen ? 1 : 0));
        CHECK(got.frozen_by == want.frozen_by);
        CHECK(got.has_t == (want.has_t ? 1 : 0));
        CHECK(got.n_staleness == want.n_staleness);
        CHECK(bits_equal(got.t, want.t));
        CHECK(bits_equal(got.worst_staleness, want.worst_staleness));
        for (int i = 0; i < AM_SKIP_COUNT; ++i) CHECK(got.skipped[i] == want.skipped[i]);
        for (int i = 0; i < 3; ++i) {
            CHECK(bits_equal(got.staleness[i], want.staleness[i]));
        }
    }
    // The fixture has to have done something, or every comparison above ran on zeros.
    CHECK(steps == 20);
    CHECK(with_frames >= 15);
    CHECK(am_pipe_g_f32_6_1_frozen(store) == 1);

    const float* pa = am_pipe_g_f32_6_1_state(store);
    const float* pb = ref.adapter().state();
    for (int i = 0; i < am_pipe_g_f32_6_1_state_size(); ++i) CHECK(pa[i] == pb[i]);

    am_guard_counters c;
    CHECK(am_pipe_g_f32_6_1_counters(store, &c) == AM_OK);
    CHECK(c.trips == 1u);
    CHECK(c.blocked_frozen > 0u);
    uint32_t mask = 0;
    CHECK(am_pipe_g_f32_6_1_frozen_by(store, &mask) == AM_OK);
    CHECK(mask == 1u);
}

void test_the_pipeline_abi_carries_lost_frames_across() {
    // Found by a mutation check: zeroing `lost_frames` in the ABI's report survived the whole
    // suite, because nothing on this side of the boundary ever overflowed a reference ring.
    // Every other field had a non-zero value somewhere and this one did not, which is the
    // quietest way for a field to go untested -- it is present, it is compared, and it is
    // compared against zero.
    //
    // `lost_frames` is the field that says input never reached the learner, so it is a poor
    // one to lose. Four-deep rings and ten arrivals between steps: six frames per step can
    // never exist.
    static float v0[4 * 1];
    static double t0[4];
    static uint8_t k0[4];
    static float v1[4 * 1];
    static double t1[4];
    static uint8_t k1[4];
    am_ring_f32 rings[2];
    CHECK(am_ring_f32_init(&rings[0], v0, t0, k0, 1, 4) == AM_OK);
    CHECK(am_ring_f32_init(&rings[1], v1, t1, k1, 1, 4) == AM_OK);

    static am_rls_storage store[AM_PIPE_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_pipe_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, AM_POLICY_HOLD_LAST,
                               0.0, 0) == AM_OK);
    for (int i = 0; i < 5; ++i) CHECK(am_pipe_f32_6_1_add_feature(store, 1, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_add_target(store, 0, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_set_constant(store, 1) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 2) == AM_OK);

    argmind::Stream<float, 1, 4> s0;
    argmind::Stream<float, 1, 4> s1;
    argmind::FrameMap<6, 1> map;
    for (int i = 0; i < 5; ++i) map.add_feature(1, 0);
    map.add_target(0, 0);
    map.set_constant(true);
    argmind::Pipeline<argmind::Rls<float, 6, 1>, 4> ref(
        argmind::Rls<float, 6, 1>(0.98f, 10.0f),
        argmind::Aligner(argmind::Policy::HoldLast, 0.0), map, 0);
    argmind::StreamView<float> views[2] = {s0.view(), s1.view()};
    CHECK(ref.configure(views, 2));

    uint32_t total_lost = 0;
    double t = 0.0;
    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 10; ++i) {
            t += 1.0;
            const float x[1] = {static_cast<float>(i) * 0.5f};
            const float y[1] = {static_cast<float>(round)};
            CHECK((am_ring_f32_push(&rings[1], t, x, 1) == AM_OK) == s1.push(t, x));
            CHECK((am_ring_f32_push(&rings[0], t, y, 1) == AM_OK) == s0.push(t, y));
        }
        am_step_report got;
        CHECK(am_pipe_f32_6_1_step(store, rings, 2, &got) == AM_OK);
        views[0] = s0.view();
        views[1] = s1.view();
        const argmind::StepReport<4> want = ref.step(views);
        CHECK(got.frames == want.frames);
        CHECK(got.lost_frames == want.lost_frames);
        CHECK(got.updated == want.updated);
        total_lost += got.lost_frames;
    }
    // The fixture has to actually lose frames, or this is another comparison against zero.
    CHECK(total_lost == 30u);
    // ...and never more than the ring can hold per step, which is the other way the
    // arithmetic could be wrong.
    CHECK(am_pipe_f32_6_1_frozen(store) == 0);
}

void test_changing_the_map_forces_a_reconfigure() {
    // The hazard this closes: `configure()` bounds-checks every map entry against the width
    // of the stream it names. A caller that widened a component afterwards would have a
    // pipeline that still says it is configured and reads off the end of a ring on the next
    // step. Mutating the map clears the flag, and `step()` refuses until it is set again.
    static float v0[8];
    static double t0[8];
    static uint8_t k0[8];
    am_ring_f32 rings[1];
    CHECK(am_ring_f32_init(&rings[0], v0, t0, k0, 1, 8) == AM_OK);

    static am_rls_storage store[AM_PIPE_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];
    CHECK(am_pipe_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f,
                               AM_POLICY_HOLD_LAST, 0.0, 0) == AM_OK);
    am_step_report rep;
    CHECK(am_pipe_f32_6_1_step(store, rings, 1, &rep) == AM_ERR_UNCONFIGURED);

    for (int i = 0; i < 5; ++i) CHECK(am_pipe_f32_6_1_add_feature(store, 0, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_add_target(store, 0, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_set_constant(store, 1) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_OK);
    CHECK(am_pipe_f32_6_1_step(store, rings, 1, &rep) == AM_OK);

    CHECK(am_pipe_f32_6_1_set_constant(store, 0) == AM_OK);   // the map moved
    CHECK(am_pipe_f32_6_1_step(store, rings, 1, &rep) == AM_ERR_UNCONFIGURED);
    // ...and it now describes five features for a six-feature adapter, so re-configuring
    // fails rather than quietly leaving x[5] holding whatever was there.
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_ERR_CONFIG);
}

void test_the_pipeline_abi_refuses_at_the_boundary() {
    static float v0[8];
    static double t0[8];
    static uint8_t k0[8];
    am_ring_f32 rings[1];
    CHECK(am_ring_f32_init(&rings[0], v0, t0, k0, 1, 8) == AM_OK);
    static am_rls_storage store[AM_PIPE_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];

    CHECK(am_pipe_f32_6_1_init(nullptr, 64, 0.98f, 10.0f, AM_POLICY_HOLD_LAST, 0.0, 0) ==
          AM_ERR_NULL);
    CHECK(am_pipe_f32_6_1_init(store, 8, 0.98f, 10.0f, AM_POLICY_HOLD_LAST, 0.0, 0) ==
          AM_ERR_TOO_SMALL);
    CHECK(am_pipe_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, 99, 0.0, 0) ==
          AM_ERR_CONFIG);
    CHECK(am_pipe_f32_6_1_init(store, sizeof(store), 0.0f, 10.0f, AM_POLICY_HOLD_LAST,
                               0.0, 0) == AM_ERR_CONFIG);

    // A non-causal policy inside a causal loop is a correctness bug, not a tuning choice, so
    // it has to be said out loud even though the aligner alone permits it.
    CHECK(am_pipe_f32_6_1_init(store, sizeof(store), 0.98f, 10.0f, AM_POLICY_INTERPOLATE,
                               0.0, 0) == AM_OK);
    for (int i = 0; i < 5; ++i) CHECK(am_pipe_f32_6_1_add_feature(store, 0, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_add_target(store, 0, 0) == AM_OK);
    CHECK(am_pipe_f32_6_1_set_constant(store, 1) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_ERR_CONFIG);
    CHECK(am_pipe_f32_6_1_set_allow_non_causal(store, 1) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_OK);

    // A staleness limit of zero. `Aligner` folds "no limit" and "zero" into one sentinel and
    // that is recorded as a real divergence; the pipeline's own gate does not, so a caller
    // meaning zero-tolerance is refused rather than silently given no check.
    CHECK(am_pipe_f32_6_1_set_max_stale_s(store, 1, 0.0) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_ERR_CONFIG);
    CHECK(am_pipe_f32_6_1_set_max_stale_s(store, 0, 0.0) == AM_OK);
    CHECK(am_pipe_f32_6_1_configure(store, rings, 1) == AM_OK);

    // Too many streams, a null ring array, and a ring whose buffers were never set.
    am_ring_f32 many[AM_MAX_STREAMS + 1];
    for (int i = 0; i <= AM_MAX_STREAMS; ++i) many[i] = rings[0];
    CHECK(am_pipe_f32_6_1_configure(store, many, AM_MAX_STREAMS + 1) == AM_ERR_CONFIG);
    CHECK(am_pipe_f32_6_1_configure(store, nullptr, 1) == AM_ERR_CONFIG);
    am_ring_f32 empty;
    empty.values = nullptr;
    empty.times = nullptr;
    empty.valid = nullptr;
    empty.dim = 1;
    empty.capacity = 8;
    empty.head = 0;
    empty.count = 0;
    empty.dropped = 0;
    empty.last_t = 0.0;
    CHECK(am_pipe_f32_6_1_configure(store, &empty, 1) == AM_ERR_NULL);

    am_step_report rep;
    CHECK(am_pipe_f32_6_1_step(nullptr, rings, 1, &rep) == AM_ERR_NULL);
    CHECK(am_pipe_f32_6_1_step(store, rings, 1, nullptr) == AM_ERR_NULL);
    CHECK(am_pipe_f32_6_1_state(nullptr) == nullptr);
}

void test_the_declared_pipeline_bounds_are_bounds_and_not_fantasies() {
    CHECK(am_pipe_f32_6_1_bytes() <= AM_PIPE_F32_6_1_MAX_BYTES);
    CHECK(am_pipe_f64_6_1_bytes() <= AM_PIPE_F64_6_1_MAX_BYTES);
    CHECK(am_pipe_g_f32_6_1_bytes() <= AM_PIPE_G_F32_6_1_MAX_BYTES);
    CHECK(am_pipe_g_f32_6_3_bytes() <= AM_PIPE_G_F32_6_3_MAX_BYTES);
    CHECK(am_pipe_g_f32_12_1_bytes() <= AM_PIPE_G_F32_12_1_MAX_BYTES);
    CHECK(am_pipe_g_f64_6_1_bytes() <= AM_PIPE_G_F64_6_1_MAX_BYTES);
    // Tight enough to budget from, as for the guard: a bound that is merely true would say
    // "reserve 100 KB" forever.
    CHECK(am_pipe_f32_6_1_bytes() * 2 >= AM_PIPE_F32_6_1_MAX_BYTES);
    CHECK(am_pipe_g_f32_6_1_bytes() * 2 >= AM_PIPE_G_F32_6_1_MAX_BYTES);
    CHECK(am_pipe_g_f32_12_1_bytes() * 2 >= AM_PIPE_G_F32_12_1_MAX_BYTES);

    // A guarded pipeline is its guard plus the pipeline's own parts, which is the
    // decomposition a caller weighing rollback depth needs.
    CHECK(am_pipe_g_f32_6_1_bytes() > am_guard_f32_6_1_bytes());
    CHECK(am_pipe_g_f32_6_1_bytes() - am_guard_f32_6_1_bytes() < 600u);
    CHECK(am_pipe_g_f32_6_1_state_size() == AM_RLS_F32_6_1_STATE);
    CHECK(am_pipe_f32_6_1_n_features() == 6);
}

}  // namespace

int main() {
    test_finiteness_predicates();
    test_row_major_matches_the_declared_layout();
    test_symmetrise_averages_pairs_and_leaves_the_diagonal();
    test_trace_accumulates_in_double_even_at_float_storage();

    test_state_is_one_contiguous_block_theta_then_p();
    test_it_recovers_known_parameters();
    test_a_posteriori_error_never_exceeds_a_priori();
    test_nonfinite_input_is_rejected_and_leaves_the_state_untouched();
    test_an_indefinite_covariance_is_refused_not_propagated();
    test_load_state_refuses_a_block_that_is_not_finite();
    test_configuration_is_reported_invalid_rather_than_thrown();
    test_memory_horizon_is_infinite_at_lambda_one();
    test_multi_output_matches_independent_single_output_instances();

    test_the_declared_size_bounds_are_bounds();
    test_the_abi_rejects_null_and_undersized_storage();
    test_the_abi_carries_the_report_and_the_residual();
    test_the_abi_round_trips_a_state_block();
    test_reset_returns_to_the_construction_state();

    test_bound_treats_non_finite_as_unhealthy();
    test_a_bound_that_cannot_fail_is_invalid();
    test_a_guard_with_no_checks_is_invalid();
    test_a_recovery_configuration_that_could_never_recover_is_invalid();
    test_an_invalid_divergence_configuration_is_invalid();
    test_bounds_beyond_capacity_are_refused_rather_than_dropped();
    test_load_state_re_arms_the_guard();
    test_manual_freeze_stops_adaptation_without_rolling_back();
    test_a_manual_freeze_mid_cadence_reports_the_live_covariance();
    test_the_checkpoint_memory_is_declared_and_matches_the_ring();
    test_a_rejected_sample_is_not_a_guard_trip();
    test_a_second_trip_does_not_roll_forward();

    test_crc32_matches_the_known_vectors();
    test_the_byte_accessors_are_little_endian_whatever_the_host_is();
    test_pack_refuses_a_buffer_that_is_too_small();
    test_pack_clamps_an_overlong_cause();
    test_unpack_reports_each_failure_distinctly();
    test_read_state_round_trips_and_refuses_a_mismatched_count();
    test_select_newest_and_next_slot();

    test_stream_push_refuses_what_the_reference_refuses();
    test_stream_overwrites_and_counts();
    test_the_ring_index_survives_wrapping();
    test_nearest_resolves_a_tie_to_the_earlier_sample();
    test_bracketing_refuses_to_extrapolate();
    test_only_interpolation_is_bounded_by_the_blend_buffer();
    test_a_missing_sample_and_a_too_wide_stream_are_distinguishable();
    test_the_causal_flag_names_the_policies_that_look_ahead();
    test_drop_if_stale_is_the_only_policy_that_refuses();

    test_a_frame_map_must_fill_the_adapter_exactly();
    test_configure_refuses_what_the_reference_raises_on();
    test_an_unconfigured_step_reports_that_rather_than_idling();
    test_a_view_whose_width_changed_is_refused();
    test_a_partial_frame_is_not_predictable_from();
    test_predicting_consumes_nothing();
    test_the_guarded_and_unguarded_forms_are_both_constructible();
    test_a_step_with_nothing_to_visit_touches_no_counter();
    test_the_declared_pipeline_memory_is_what_it_says();

    test_the_guard_abi_agrees_with_the_header_bit_for_bit();
    test_the_guard_abi_refuses_at_the_boundary_rather_than_trusting();
    test_the_declared_guard_bounds_are_bounds_and_not_fantasies();
    test_the_record_abi_round_trips_and_arbitrates();
    test_the_abi_closes_the_loop_from_a_guard_to_a_slot_and_back();

    test_the_caller_owned_ring_behaves_exactly_like_the_typed_one();
    test_the_pipeline_abi_agrees_with_the_header_bit_for_bit();
    test_the_pipeline_abi_carries_lost_frames_across();
    test_changing_the_map_forces_a_reconfigure();
    test_the_pipeline_abi_refuses_at_the_boundary();
    test_the_declared_pipeline_bounds_are_bounds_and_not_fantasies();
    test_the_declared_ring_memory_matches_the_shape();

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
