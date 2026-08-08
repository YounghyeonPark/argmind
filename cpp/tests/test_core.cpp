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
    test_the_declared_ring_memory_matches_the_shape();

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
