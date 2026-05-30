// Copyright, 2026. See LICENSE for copyright holder information.
//
// ============================================================================
// SPARSE-DENSE MATRIX MULTIPLICATION PERFORMANCE BENCHMARK
// ============================================================================
//
// Benchmarks the RandBLAS left_spmm kernels (C = A*B, with A sparse) across the
// three sparse formats (CSR, CSC, COO) and BOTH dense-matrix layouts
// (ColMajor, RowMajor). Driving left_spmm with both layouts is enough to
// exercise every hand-rolled SpMM kernel, because the dispatch
// (spmm_dispatch.hh) selects the kernel purely from (layout_opB, layout_C):
//
//   ColMajor dense -> apply_csr_jik_p11          / apply_csc_jki_p11
//   RowMajor dense -> apply_csr_ikb_p1b_rowmajor / apply_csc_kib_1p1_rowmajor
//   (COO delegates to the CSC kernels in both layouts, via apply_coo_via_csc)
//
// A fourth combination -- op(B)=Trans (computing C = A * B^T, the dense operand
// fed transposed) -- also routes to apply_csr_jik_p11 / apply_csc_jki_p11, but
// with a STRIDED dense access (gather when C is ColMajor, strided store when C
// is RowMajor) rather than the unit-stride NoTrans ColMajor pattern. MKL
// declines op(B)=Trans, so it is hand-rolled even on MKL builds.
//
// These are exactly the kernels that left_spmm and right_spmm TOGETHER reach.
// right_spmm reduces to left_spmm by transposing the sparse operand (CSR<->CSC
// view) and flipping the layout, so the old "right CSR (ColMajor)" path runs
// the very same kernel as "left RowMajor CSC" does here (and old "right CSC" ==
// "left RowMajor CSR"). Benchmarking left_spmm directly in both layouts hits
// the same kernels while measuring each one without the right_spmm wrapper.
//
// This benchmark answers:
//   1. Which sparse format is fastest, for each dense layout?
//   2. ColMajor vs RowMajor: how much does the dense layout matter? RowMajor
//      routes to the BLAS-axpy-based kernels; ColMajor routes to the scalar
//      gather (CSR) / scatter (CSC) kernels.
//   3. (MKL builds only) How much does MKL accelerate over the hand-rolled
//      kernels? The ColMajor table runs CSR through both left_spmm (which uses
//      MKL when available) and the hand-rolled kernel called directly. This is
//      the one path that does NOT go through left_spmm -- on an MKL build the
//      dispatch always picks MKL, so the hand-rolled kernel is otherwise
//      unreachable. It compiles out entirely when MKL is disabled.
//   4. op(B)=Trans (C = A * B^T): how much does the strided dense access cost
//      relative to the unit-stride NoTrans path? Reported in two extra tables.
//
// NOTE: benchmarks the SpMM kernel directly, not sketch_general. SpMM dominates
// sketching cost, so these results are a good proxy for sketching performance.
//
// NOTATION:
//   A - sparse (m x n),  B - dense (n x d),  C - dense result (m x d)
//   left_spmm:  C(m x d) = A(m x n) * B(n x d)
//   The same logical B (and hence result C) is stored in BOTH ColMajor and
//   RowMajor, so the two layouts compute the identical product and are
//   directly comparable.
//
// USAGE:
//   ./spmm_performance                             # default sweep
//   ./spmm_performance m n d density [num_trials]  # single config
//
//   Default sweep (no arguments):
//     Two sections, 10 trials each:
//       1. SQUARE (d = m = n): sizes 100..2000
//       2. RECTANGULAR: square A with small d, tall A, wide A
//
//   Single config (4+ arguments):
//     One (m, n, d, density) configuration, default 20 trials.
//
// EXAMPLES:
//   env OMP_NUM_THREADS=8 ./spmm_performance
//   env OMP_NUM_THREADS=8 ./spmm_performance 2000 2000 200 0.01
//   env OMP_NUM_THREADS=8 ./spmm_performance 5000 500 200 0.01 10
//
// ============================================================================

#include <RandBLAS.hh>
#include <blas.hh>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Internal headers for sparse dispatch, format conversions, and generation
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"
#include "RandBLAS/testing/sparse_data.hh"

#include "RandBLAS/config.h"

using namespace std::chrono;
using blas::Layout;
using blas::Op;

// Calls the hand-rolled CSR left-multiply kernel directly, bypassing MKL.
// Used to answer question 3 (MKL vs hand-rolled speedup) on MKL builds, where
// left_spmm would otherwise always pick MKL. Replicates the beta-scaling and
// kernel selection logic from spmm_dispatch.hh. Here A is sparse (CSR), B is the
// dense input, C is the dense result -- matching the left_spmm convention.
template <typename T, typename sint_t>
void handrolled_left_spmm_csr(
    blas::Layout layout, int64_t d, int64_t n, int64_t m,
    T alpha, const RandBLAS::sparse_data::CSRMatrix<T, sint_t> &A,
    const T *B, int64_t ldb, T beta, T *C, int64_t ldc
) {
    // Apply beta to C (same as dispatch)
    if (layout == Layout::ColMajor) {
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &C[i*ldc]);
    } else {
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &C[i*ldc]);
    }
    if (alpha == (T)0) return;

    // Call the hand-rolled CSR kernel directly (same as dispatch fallback)
    Layout layout_opB = layout;  // opB == NoTrans
    Layout layout_C = layout;
    if (layout_opB == Layout::RowMajor && layout_C == Layout::RowMajor) {
        RandBLAS::sparse_data::csr::apply_csr_ikb_p1b_rowmajor(
            alpha, d, n, m, A, B, ldb, C, ldc);
    } else {
        RandBLAS::sparse_data::csr::apply_csr_jik_p11(
            alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
    }
}

// Run num_trials repetitions, return {min, median} times in microseconds.
template <typename Func>
std::pair<long, long> run_trials(Func&& func, int num_trials) {
    std::vector<long> times;
    times.reserve(num_trials);

    for (int t = 0; t < num_trials; ++t) {
        auto start = steady_clock::now();
        func();
        auto end = steady_clock::now();
        times.push_back(duration_cast<microseconds>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    return {times[0], times[num_trials / 2]};
}

// Like run_trials, but times densify and compute phases separately.
// Returns {min_densify, min_compute}.
template <typename DensifyFunc, typename ComputeFunc>
std::pair<long, long> run_split_trials(
    DensifyFunc&& densify, ComputeFunc&& compute, int num_trials
) {
    std::vector<long> t_dens, t_comp;
    t_dens.reserve(num_trials);
    t_comp.reserve(num_trials);

    for (int t = 0; t < num_trials; ++t) {
        auto s1 = steady_clock::now();
        densify();
        auto s2 = steady_clock::now();
        compute();
        auto s3 = steady_clock::now();
        t_dens.push_back(duration_cast<microseconds>(s2 - s1).count());
        t_comp.push_back(duration_cast<microseconds>(s3 - s2).count());
    }

    std::sort(t_dens.begin(), t_dens.end());
    std::sort(t_comp.begin(), t_comp.end());
    return {t_dens[0], t_comp[0]};
}

// Output formatting helpers.
void print_row(const std::string& name, long min_us, long med_us, long baseline) {
    double ratio = (double)min_us / baseline;
    std::cout << "  " << std::setw(24) << std::left << name
              << std::setw(10) << std::right << min_us
              << std::setw(10) << med_us
              << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x\n";
}

void print_densify_row(const std::string& name, long total, long dens, long gemm, long baseline) {
    double ratio = (double)total / baseline;
    std::cout << "  " << std::setw(24) << std::left << name
              << std::setw(10) << std::right << total
              << std::setw(10) << total
              << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x"
              << "  (densify " << dens << " + GEMM " << gemm << ")\n";
}

void print_table_header(const std::string& title) {
    std::cout << "  " << title << "\n";
    std::cout << "  " << std::setw(24) << std::left << "Kernel"
              << std::setw(10) << std::right << "Min (us)"
              << std::setw(10) << "Med (us)"
              << std::setw(11) << "vs best" << "\n";
    std::cout << "  " << std::string(55, '-') << "\n";
}

// Run one complete benchmark configuration: generate matrices, time every
// format x layout combination through left_spmm, verify correctness, print.
void run_config(int64_t m, int64_t n, int64_t d, double density, int num_trials) {
    using T = double;
    namespace rbsd = RandBLAS::sparse_data;
    uint64_t seed = 12345;

    // Header
    std::string shape;
    if (m == n && d == m) shape = "all square";
    else if (m == n) shape = "square A";
    else if (m > n) shape = "tall";
    else shape = "wide";

    std::cout << "--- A(" << m << "x" << n << "), d=" << d
              << ", density=" << std::setprecision(4) << density << " (" << shape << ") ---\n";

    // Generate one random COO matrix and convert to CSR/CSC so all three
    // formats represent the identical mathematical matrix.
    auto [A_coo, next_state] = RandBLAS::testing::random_coo<T>(m, n, density, RandBLAS::RNGState<>(seed));
    auto A_csr = A_coo.as_owning_csr();
    auto A_csc = A_coo.as_owning_csc();

    std::cout << "  nnz=" << A_coo.nnz << ", trials=" << num_trials << "\n\n";

    // One logical dense B (n x d), stored in both layouts so the ColMajor and
    // RowMajor runs compute the identical product C = A*B.
    //   ColMajor: B(i,j) = B_cm[i + j*n]   (ldB = n)
    //   RowMajor: B(i,j) = B_rm[i*d + j]   (ldB = d)
    std::vector<T> B_cm(n * d);
    RandBLAS::DenseDist DB(n, d);
    RandBLAS::fill_dense(DB, B_cm.data(), next_state);
    std::vector<T> B_rm(n * d);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < d; ++j)
            B_rm[i*d + j] = B_cm[i + j*n];

    // Result buffers (logical C is m x d in both layouts) and densify workspace.
    std::vector<T> C_cm(m * d), C_rm(m * d);
    std::vector<T> A_dense(m * n);

    // Time left_spmm for one (format, layout): C = A * B, with B and C stored
    // in `layout`. ldB / ldC follow from the layout (see notation above).
    auto time_spmm = [&](const auto& A, Layout layout, std::vector<T>& C) {
        const T* Bptr = (layout == Layout::ColMajor) ? B_cm.data() : B_rm.data();
        int64_t ldB   = (layout == Layout::ColMajor) ? n : d;
        int64_t ldC   = (layout == Layout::ColMajor) ? m : d;
        return run_trials([&]() {
            std::fill(C.begin(), C.end(), 0.0);
            rbsd::left_spmm(layout, Op::NoTrans, Op::NoTrans, m, d, n,
                            1.0, A, 0, 0, Bptr, ldB, 0.0, C.data(), ldC);
        }, num_trials);
    };

    // ---- ColMajor left_spmm (hits jik / jki kernels) ----
    auto [min_cm_csr, med_cm_csr] = time_spmm(A_csr, Layout::ColMajor, C_cm);
    auto [min_cm_csc, med_cm_csc] = time_spmm(A_csc, Layout::ColMajor, C_cm);
    auto [min_cm_coo, med_cm_coo] = time_spmm(A_coo, Layout::ColMajor, C_cm);

    // Hand-rolled CSR ColMajor: bypasses MKL for the MKL-vs-hand-rolled
    // comparison. This is the only path that does not go through left_spmm.
    #if defined(RandBLAS_HAS_MKL)
    long min_cm_csr_hr = 0, med_cm_csr_hr = 0;
    {
        auto [mn, md] = run_trials([&]() {
            std::fill(C_cm.begin(), C_cm.end(), 0.0);
            handrolled_left_spmm_csr(Layout::ColMajor, m, d, n, 1.0,
                A_csr, B_cm.data(), n, 0.0, C_cm.data(), m);
        }, num_trials);
        min_cm_csr_hr = mn;
        med_cm_csr_hr = md;
    }
    #endif

    // ---- RowMajor left_spmm (hits ikb / kib kernels; the paths the old
    //      right_spmm section reached, via the CSR<->CSC transpose) ----
    auto [min_rm_csr, med_rm_csr] = time_spmm(A_csr, Layout::RowMajor, C_rm);
    auto [min_rm_csc, med_rm_csc] = time_spmm(A_csc, Layout::RowMajor, C_rm);
    auto [min_rm_coo, med_rm_coo] = time_spmm(A_coo, Layout::RowMajor, C_rm);

    // ---- Transposed dense (opB=Trans): C = A * B^T, dense operand fed
    //      transposed so the SAME product A*B is computed. This routes to the
    //      jik/jki kernels with a STRIDED dense access (incv != 1 when C is
    //      ColMajor, incAv != 1 when C is RowMajor) -- distinct from the
    //      unit-stride NoTrans ColMajor path, and the only sparse-NoTrans path
    //      MKL declines (so it stays hand-rolled on MKL builds).
    //      B^T (d x n) needs no new storage: the transpose of the ColMajor n-x-d
    //      buffer IS the RowMajor buffer, so ColMajor-B^T == B_rm, RowMajor == B_cm.
    auto time_spmm_t = [&](const auto& A, Layout layout, std::vector<T>& C) {
        const T* BTptr = (layout == Layout::ColMajor) ? B_rm.data() : B_cm.data();
        int64_t ldBT   = (layout == Layout::ColMajor) ? d : n;
        int64_t ldC    = (layout == Layout::ColMajor) ? m : d;
        return run_trials([&]() {
            std::fill(C.begin(), C.end(), 0.0);
            rbsd::left_spmm(layout, Op::NoTrans, Op::Trans, m, d, n,
                            1.0, A, 0, 0, BTptr, ldBT, 0.0, C.data(), ldC);
        }, num_trials);
    };

    auto [min_tcm_csr, med_tcm_csr] = time_spmm_t(A_csr, Layout::ColMajor, C_cm);
    auto [min_tcm_csc, med_tcm_csc] = time_spmm_t(A_csc, Layout::ColMajor, C_cm);
    auto [min_tcm_coo, med_tcm_coo] = time_spmm_t(A_coo, Layout::ColMajor, C_cm);
    auto [min_trm_csr, med_trm_csr] = time_spmm_t(A_csr, Layout::RowMajor, C_rm);
    auto [min_trm_csc, med_trm_csc] = time_spmm_t(A_csc, Layout::RowMajor, C_rm);
    auto [min_trm_coo, med_trm_coo] = time_spmm_t(A_coo, Layout::RowMajor, C_rm);

    // ---- Reference: densify + GEMM, per layout ----
    // The ColMajor reference (ref_cm) doubles as the correctness oracle.
    std::vector<T> ref_cm(m * d);
    auto [min_dens_cm, min_gemm_cm] = run_split_trials(
        [&]() { rbsd::csr::csr_to_dense(A_csr, Layout::ColMajor, A_dense.data()); },
        [&]() {
            std::fill(ref_cm.begin(), ref_cm.end(), 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       m, d, n, 1.0, A_dense.data(), m, B_cm.data(), n, 0.0, ref_cm.data(), m);
        }, num_trials);
    long min_ref_cm = min_dens_cm + min_gemm_cm;

    auto [min_dens_rm, min_gemm_rm] = run_split_trials(
        [&]() { rbsd::csr::csr_to_dense(A_csr, Layout::RowMajor, A_dense.data()); },
        [&]() {
            std::fill(C_rm.begin(), C_rm.end(), 0.0);
            blas::gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans,
                       m, d, n, 1.0, A_dense.data(), n, B_rm.data(), d, 0.0, C_rm.data(), d);
        }, num_trials);
    long min_ref_rm = min_dens_rm + min_gemm_rm;

    // ---- Correctness: re-run each (format, layout) once, compare to ref_cm ----
    bool all_pass = true;
    int num_checks = 0;
    std::vector<T> result(m * d);

    // All NoTrans and Trans paths compute the same logical product A*B, so they
    // are all checked against the single ColMajor oracle ref_cm. For opB=Trans
    // the dense operand is B^T (== the other layout's buffer; see timing above).
    auto check = [&](const std::string& label, const auto& A, Layout layout, Op opB) {
        const T* Bptr;
        int64_t ldB;
        if (opB == Op::NoTrans) {
            Bptr = (layout == Layout::ColMajor) ? B_cm.data() : B_rm.data();
            ldB  = (layout == Layout::ColMajor) ? n : d;
        } else {
            Bptr = (layout == Layout::ColMajor) ? B_rm.data() : B_cm.data();
            ldB  = (layout == Layout::ColMajor) ? d : n;
        }
        int64_t ldC = (layout == Layout::ColMajor) ? m : d;
        std::fill(result.begin(), result.end(), 0.0);
        rbsd::left_spmm(layout, Op::NoTrans, opB, m, d, n,
                        1.0, A, 0, 0, Bptr, ldB, 0.0, result.data(), ldC);
        double maxdiff = 0;
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < d; ++j) {
                int64_t idx = (layout == Layout::ColMajor) ? (i + j*m) : (i*d + j);
                maxdiff = std::max(maxdiff, std::abs(result[idx] - ref_cm[i + j*m]));
            }
        num_checks++;
        if (maxdiff > 1e-10) {
            std::cout << "  FAIL: " << label << " max|diff|=" << std::scientific << maxdiff << "\n";
            all_pass = false;
        }
    };

    check("ColMajor+CSR",      A_csr, Layout::ColMajor, Op::NoTrans);
    check("ColMajor+CSC",      A_csc, Layout::ColMajor, Op::NoTrans);
    check("ColMajor+COO",      A_coo, Layout::ColMajor, Op::NoTrans);
    check("RowMajor+CSR",      A_csr, Layout::RowMajor, Op::NoTrans);
    check("RowMajor+CSC",      A_csc, Layout::RowMajor, Op::NoTrans);
    check("RowMajor+COO",      A_coo, Layout::RowMajor, Op::NoTrans);
    check("ColMajor+CSR(B^T)", A_csr, Layout::ColMajor, Op::Trans);
    check("ColMajor+CSC(B^T)", A_csc, Layout::ColMajor, Op::Trans);
    check("ColMajor+COO(B^T)", A_coo, Layout::ColMajor, Op::Trans);
    check("RowMajor+CSR(B^T)", A_csr, Layout::RowMajor, Op::Trans);
    check("RowMajor+CSC(B^T)", A_csc, Layout::RowMajor, Op::Trans);
    check("RowMajor+COO(B^T)", A_coo, Layout::RowMajor, Op::Trans);

    // Also verify the hand-rolled CSR path.
    #if defined(RandBLAS_HAS_MKL)
    {
        std::fill(result.begin(), result.end(), 0.0);
        handrolled_left_spmm_csr(Layout::ColMajor, m, d, n, 1.0,
            A_csr, B_cm.data(), n, 0.0, result.data(), m);
        double maxdiff = 0;
        for (int64_t i = 0; i < m * d; ++i)
            maxdiff = std::max(maxdiff, std::abs(result[i] - ref_cm[i]));
        num_checks++;
        if (maxdiff > 1e-10) {
            std::cout << "  FAIL: ColMajor+CSR(hand-rolled) max|diff|=" << std::scientific << maxdiff << "\n";
            all_pass = false;
        }
    }
    #endif

    if (all_pass)
        std::cout << "  Correctness: all " << num_checks << " checks PASS\n";
    std::cout << "\n";

    // ---- Results tables ----
    std::cout << std::fixed << std::setprecision(2);

    // ColMajor table
    #if defined(RandBLAS_HAS_MKL)
    long bcm = std::min({min_cm_csr, min_cm_csr_hr, min_cm_csc, min_cm_coo});
    #else
    long bcm = std::min({min_cm_csr, min_cm_csc, min_cm_coo});
    #endif
    print_table_header("LEFT SPMM, ColMajor dense: C(" + std::to_string(m) + "x" +
                       std::to_string(d) + ") = A * B   (jik/jki kernels)");
    #if defined(RandBLAS_HAS_MKL)
    print_row("CSR (MKL)",         min_cm_csr,    med_cm_csr,    bcm);
    print_row("CSR (hand-rolled)", min_cm_csr_hr, med_cm_csr_hr, bcm);
    #else
    print_row("CSR",               min_cm_csr,    med_cm_csr,    bcm);
    #endif
    print_row("CSC",               min_cm_csc, med_cm_csc, bcm);
    print_row("COO",               min_cm_coo, med_cm_coo, bcm);
    print_densify_row("densify+GEMM", min_ref_cm, min_dens_cm, min_gemm_cm, bcm);
    std::cout << "\n";

    // RowMajor table
    long brm = std::min({min_rm_csr, min_rm_csc, min_rm_coo});
    print_table_header("LEFT SPMM, RowMajor dense: C(" + std::to_string(m) + "x" +
                       std::to_string(d) + ") = A * B   (ikb/kib kernels)");
    print_row("CSR", min_rm_csr, med_rm_csr, brm);
    print_row("CSC", min_rm_csc, med_rm_csc, brm);
    print_row("COO", min_rm_coo, med_rm_coo, brm);
    print_densify_row("densify+GEMM", min_ref_rm, min_dens_rm, min_gemm_rm, brm);
    std::cout << "\n";

    // Transposed-dense tables (opB=Trans -> jik/jki with strided dense access).
    // MKL declines opB=Trans, so these stay hand-rolled even on MKL builds.
    long btc = std::min({min_tcm_csr, min_tcm_csc, min_tcm_coo});
    print_table_header("LEFT SPMM, transposed dense, ColMajor C: C = A * B^T   (opB=Trans -> jik/jki, strided gather)");
    print_row("CSR", min_tcm_csr, med_tcm_csr, btc);
    print_row("CSC", min_tcm_csc, med_tcm_csc, btc);
    print_row("COO", min_tcm_coo, med_tcm_coo, btc);
    std::cout << "\n";

    long btr = std::min({min_trm_csr, min_trm_csc, min_trm_coo});
    print_table_header("LEFT SPMM, transposed dense, RowMajor C: C = A * B^T   (opB=Trans -> jik/jki, strided store)");
    print_row("CSR", min_trm_csr, med_trm_csr, btr);
    print_row("CSC", min_trm_csc, med_trm_csc, btr);
    print_row("COO", min_trm_coo, med_trm_coo, btr);
    std::cout << "\n";

    // Summary: best NoTrans (ColMajor vs RowMajor) and the transposed-dense cost.
    long bnt = std::min(bcm, brm);
    long bt  = std::min(btc, btr);
    std::cout << "  SUMMARY: best NoTrans ColMajor " << bcm << " us  |  best NoTrans RowMajor " << brm << " us  |  ";
    if (brm <= bcm)
        std::cout << "RowMajor " << std::fixed << std::setprecision(2) << (double)bcm / brm << "x faster\n";
    else
        std::cout << "ColMajor " << std::fixed << std::setprecision(2) << (double)brm / bcm << "x faster\n";
    std::cout << "           best transposed-dense (opB=Trans) " << bt << " us  ("
              << std::fixed << std::setprecision(2) << (double)bt / bnt << "x vs best NoTrans)\n\n";
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "============================================================================\n";
    std::cout << "SPMM PERFORMANCE BENCHMARK\n";
    std::cout << "============================================================================\n";
#if defined(RandBLAS_HAS_MKL)
    std::cout << "MKL support: ENABLED\n";
#else
    std::cout << "MKL support: DISABLED\n";
#endif
    std::cout << "\n";
    std::cout << "  A is m-by-n (sparse), B and C are dense.\n";
    std::cout << "  left_spmm:  C(m x d) = A(m x n) * B(n x d)\n";
    std::cout << "  All kernels are reached through left_spmm; the ColMajor and\n";
    std::cout << "  RowMajor dense layouts select the two kernel families.\n\n";

    if (argc >= 5) {
        // Single config mode
        int64_t m = std::atoll(argv[1]);
        int64_t n = std::atoll(argv[2]);
        int64_t d = std::atoll(argv[3]);
        double density = std::atof(argv[4]);
        int num_trials = (argc > 5) ? std::atoi(argv[5]) : 20;
        run_config(m, n, d, density, num_trials);
    } else {
        // Default sweep
        struct Config { int64_t m, n, d; double density; };
        int num_trials = 10;

        // Section 1: Square problems (d = m = n)
        std::vector<Config> square_configs = {
            {  100,   100,   100, 0.01},
            {  200,   200,   200, 0.01},
            {  500,   500,   500, 0.01},
            { 1000,  1000,  1000, 0.01},
            { 2000,  2000,  2000, 0.01},
        };

        std::cout << "=== SQUARE PROBLEMS (d = m = n) ===\n";
        std::cout << num_trials << " trials per config\n\n";
        for (auto& c : square_configs) {
            run_config(c.m, c.n, c.d, c.density, num_trials);
        }

        // Section 2: Rectangular problems
        std::vector<Config> rect_configs = {
            { 2000,  2000,   200, 0.01},
            { 5000,  5000,   500, 0.001},
            { 5000,   500,   500, 0.01},
            {  500,  5000,   500, 0.01},
        };

        std::cout << "=== RECTANGULAR PROBLEMS ===\n";
        std::cout << num_trials << " trials per config\n\n";
        for (auto& c : rect_configs) {
            run_config(c.m, c.n, c.d, c.density, num_trials);
        }
    }

    return 0;
}
