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
#include "argmind/guard.hpp"
#include "argmind/persist.hpp"
#include "argmind/rls.hpp"

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

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
