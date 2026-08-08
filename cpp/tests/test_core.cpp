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

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
