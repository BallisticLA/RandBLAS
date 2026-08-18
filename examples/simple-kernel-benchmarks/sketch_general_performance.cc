// Copyright, 2026. See LICENSE for copyright holder information.
//
// ============================================================================
// SKETCH_GENERAL PERFORMANCE BENCHMARK (sparse sketching operators)
// ============================================================================
//
// Analogous to spmm_performance.cc, but it benchmarks the full sketching call
// sketch_general() with a real SparseSkOp -- not the low-level left_spmm on a
// generic uniform-random matrix. This exercises the *structured* operators that
// spmm_performance never produces:
//
//   * Wide SASO   SparseDist(d, m, k, Short)  -- fixed k nnz per column
//   * CountSketch SparseDist(d, m, 1, Short)  -- exactly one nnz per column
//   * Wide LASO   SparseDist(d, m, k, Long)   -- <= k nnz per row, empty cols
//
// and (via the right-sketch / tall operators) the transpose reduction that maps
// a tall SASO right-sketch onto the same CSC kernel path as a wide SASO.
//
// WHY WORK-NORMALIZED METRICS: raw wall-time conflates "how much work" with "how
// efficiently it's done". A vec_nnz=8 operator has 4x the nonzeros of vec_nnz=2,
// and a wide SASO has ~m/d x the nonzeros of a wide LASO at the same vec_nnz, so
// raw times are not comparable across those axes. We therefore report:
//
//   * ns/elt = time / (nnz * work_mult): nanoseconds per nonzero-output-element.
//     work_mult = n for a left-sketch (each S-nonzero touches a length-n output
//     row), m for a right-sketch. This IS comparable across vec_nnz and SASO/LASO;
//     for an appropriately optimized kernel it stays roughly flat as vec_nnz grows.
//   * GB/s (%STR) = a MODEL byte count / time, as a fraction of measured STREAM
//     bandwidth. These kernels are memory-bound (arithmetic intensity ~ vec_nnz/4
//     flop/byte), so % of STREAM is the roofline ceiling: ~80% means "done", ~20%
//     means headroom. The byte count is an idealized-reuse model, not a hardware
//     counter (perf counters are SIP-gated on macOS) -- read it as a bound. Two
//     caveats: (a) A-read traffic is bounded by the DISTINCT rows/cols of A actually
//     touched (<= min(nnz, contract_dim)), so a sparse LASO reads far less than all
//     of A; (b) STREAM is a DRAM ceiling, so a small cache-resident problem (e.g. a
//     tiny LASO) can legitimately exceed 100% %STR -- that flags "fits in cache",
//     not a measurement error.
//
// Parallel efficiency needs the SAME kernel run at multiple thread counts, so it
// is NOT free per run; it lives behind --scaling (see USAGE).
//
// The cost of a sparse sketch has three phases; we time them separately:
//   1. SAMPLE   fill_sparse(S): populate (rows, cols, vals).
//   2. CONVERT  the COO->CSC sort inside apply_coo_via_csx (deepcopy + re-sort,
//               incurred every apply when the operator's COO is not CSC-sorted).
//   3. KERNEL   apply_csc_jki_p11 (ColMajor) / apply_csc_kib_1p1_rowmajor (RowMajor).
// WARM apply (S pre-sampled; phases 2+3) is the primary number; COLD (1+2+3) and
// CONVERT-only appear in the notes of the left/ColMajor table.
//
// NOTATION (left-sketch):   B(d x n) = alpha * S(d x m) * A(m x n)
//          (right-sketch):  B(m x d) = alpha * A(m x n) * S(n x d)
//
// USAGE:
//   ./sketch_general_performance [flags]                          # default sweep
//   ./sketch_general_performance [flags] d m n vec_nnz major [trials]  # single cfg
//       major: 0 = Short (SASO), 1 = Long (LASO)
//   flags:
//     --no-stream            skip the startup STREAM calibration (%STR prints '-')
//     --scaling              thread-scaling mode: speedup/efficiency/%STR per thread
//     --threads=1,2,4,8      thread counts for --scaling (default 1,2,4,8)
//
// EXAMPLES:
//   env OMP_NUM_THREADS=4 ./sketch_general_performance
//   env OMP_NUM_THREADS=4 ./sketch_general_performance 200 2000 2000 4 0
//   ./sketch_general_performance --scaling --threads=1,2,4,8 200 2000 2000 4 0
//
// ============================================================================

#include <RandBLAS.hh>
#include <blas.hh>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#include "RandBLAS/config.h"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

using namespace std::chrono;
using blas::Layout;
using blas::Op;
using RandBLAS::Axis;
using RandBLAS::SparseDist;
using RandBLAS::SparseSkOp;

// Machine bandwidth ceiling (GB/s) used as the %STR denominator; <0 disables %STR.
static double g_stream_gbps = -1.0;

// ---- OpenMP shims (no-ops without OpenMP) ----------------------------------
static int current_threads() {
#if defined(RandBLAS_HAS_OpenMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static void set_threads(int t) {
#if defined(RandBLAS_HAS_OpenMP)
    omp_set_dynamic(0);
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}

static int effective_threads(int requested_threads) {
#if defined(RandBLAS_HAS_OpenMP)
    int actual_threads = 1;
    #pragma omp parallel num_threads(requested_threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
    }
    return actual_threads;
#else
    (void) requested_threads;
    return 1;
#endif
}

class OpenMPSettingsGuard {
public:
    OpenMPSettingsGuard() {
#if defined(RandBLAS_HAS_OpenMP)
        dynamic_ = omp_get_dynamic();
        threads_ = omp_get_max_threads();
#endif
    }

    ~OpenMPSettingsGuard() {
#if defined(RandBLAS_HAS_OpenMP)
        omp_set_num_threads(threads_);
        omp_set_dynamic(dynamic_);
#endif
    }

private:
#if defined(RandBLAS_HAS_OpenMP)
    int dynamic_;
    int threads_;
#endif
};

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

// STREAM triad (a = b + scalar*c) at the current OpenMP thread count. Returns
// achievable bandwidth in GB/s (min time over reps; 3*N*8 bytes per pass).
static double measure_stream_gbps(int64_t N = (int64_t)1 << 24, int reps = 10) {
    std::vector<double> a(N), b(N), c(N);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
    const double scalar = 3.0;
    // warmup
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) a[i] = b[i] + scalar * c[i];
    auto best = std::numeric_limits<microseconds::rep>::max();
    for (int r = 0; r < reps; ++r) {
        auto s = steady_clock::now();
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) a[i] = b[i] + scalar * c[i];
        auto e = steady_clock::now();
        best = std::min(best, duration_cast<microseconds>(e - s).count());
    }
    volatile double sink = a[N - 1]; (void)sink;       // defeat dead-store elision
    double bytes = 3.0 * (double)N * sizeof(double);
    return best > 0 ? bytes / (double)best / 1000.0 : -1.0; // bytes/us/1000 == GB/s
}

// ---- per-row metrics -------------------------------------------------------
struct Row {
    std::string label;
    long   min_us = 0, med_us = 0;
    double ns_per_elt = -1; // time / (nnz * work_mult)
    double gbps       = -1; // model bytes / time
    double pct_stream = -1; // gbps / g_stream_gbps
    std::string notes;
};

// work_mult = output cols touched per nonzero (n for left-sketch, m for right).
// read_dense = elements of A read; out_elems = elements of the dense output.
static void fill_metrics(Row& r, long min_us, int64_t nnz, int64_t work_mult,
                         int64_t read_dense, int64_t out_elems) {
    int64_t work = nnz * work_mult;
    r.ns_per_elt = (work > 0) ? (double)min_us * 1000.0 / (double)work : -1;
    double bytes = (double)(read_dense + 2 * out_elems) * sizeof(double)
                 + (double)nnz * (sizeof(double) + sizeof(int64_t));
    r.gbps = (min_us > 0) ? bytes / (double)min_us / 1000.0 : -1;
    r.pct_stream = (g_stream_gbps > 0 && r.gbps > 0) ? r.gbps / g_stream_gbps * 100.0 : -1;
}

static std::string fcell(double v, int prec) {
    if (v < 0) return "-";
    std::ostringstream o; o << std::fixed << std::setprecision(prec) << v; return o.str();
}

static void print_metric_header(const std::string& title) {
    std::cout << "  " << title << "\n";
    std::cout << "  " << std::left << std::setw(26) << "Operator" << std::right
              << std::setw(9) << "Min(us)" << std::setw(9) << "Med(us)"
              << std::setw(9) << "ns/elt" << std::setw(8) << "GB/s"
              << std::setw(7) << "%STR" << "  notes\n";
    std::cout << "  " << std::string(76, '-') << "\n";
}

static void print_metric_row(const Row& r) {
    std::cout << "  " << std::left << std::setw(26) << r.label << std::right
              << std::setw(9) << r.min_us << std::setw(9) << r.med_us
              << std::setw(9) << fcell(r.ns_per_elt, 3)
              << std::setw(8) << fcell(r.gbps, 1)
              << std::setw(7) << fcell(r.pct_stream, 0)
              << "  " << r.notes << "\n";
}

static const char* sort_name(RandBLAS::sparse_data::NonzeroSort s) {
    using NS = RandBLAS::sparse_data::NonzeroSort;
    switch (s) {
        case NS::CSC:  return "CSC";
        case NS::CSR:  return "CSR";
        default:       return "None";
    }
}

// One distribution config, benchmarked across both layouts on one side.
struct OpSpec {
    std::string label;
    int64_t vec_nnz;
    Axis     axis;
};

struct SamplingScalingRow {
    int requested_threads;
    int threads;
    long min_us;
    long median_us;
    double ns_per_nonzero;
    double speedup;
    double efficiency;
};

// ---------------------------------------------------------------------------
// LEFT-SKETCH:  B(d x n) = S(d x m) * A(m x n),  S ~ SparseDist(d, m, k, axis).
// work_mult = n; read_dense(A) = m*n; out_elems(B) = d*n.
// ---------------------------------------------------------------------------
void run_left(int64_t d, int64_t m, int64_t n,
              const std::vector<OpSpec>& specs, int num_trials) {
    using T = double;
    namespace rb = RandBLAS;
    uint64_t seed = 12345;

    std::cout << "=== LEFT-SKETCH  B(" << d << "x" << n << ") = S(" << d << "x" << m
              << ") * A(" << m << "x" << n << "),  trials=" << num_trials << " ===\n\n";

    std::vector<T> A_cm(m * n), A_rm(m * n);
    rb::DenseDist DA(m, n);
    auto st = rb::fill_dense(DA, A_cm.data(), rb::RNGState<>(seed));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            A_rm[i * n + j] = A_cm[i + j * m];

    std::vector<T> B_cm(d * n), B_rm(d * n), B_ref(d * n), result(d * n);
    std::vector<T> S_dense(d * m);

    std::vector<Row> rows_cm, rows_rm;
    Row oracle; oracle.label = "densify(S)+GEMM"; oracle.notes = "dense oracle";

    for (const auto& spec : specs) {
        SparseDist dist(d, m, spec.vec_nnz, spec.axis);
        SparseSkOp<T> S(dist, st);
        rb::fill_sparse(S);
        auto view = rb::sparse::coo_view_of_skop(S);
        int64_t nnz = S.nnz;
        std::string base_note = std::string(sort_name(view.sort)) + " nnz=" + std::to_string(nnz);

        // Oracle: densify S then GEMM (ColMajor); also the correctness oracle.
        rb::sparse_data::coo::coo_to_dense(view, Layout::ColMajor, S_dense.data());
        auto [t_gemm, m_gemm] = run_trials([&]() {
            std::fill(B_ref.begin(), B_ref.end(), 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                       1.0, S_dense.data(), d, A_cm.data(), m, 0.0, B_ref.data(), d);
        }, num_trials);
        oracle.min_us = t_gemm; oracle.med_us = m_gemm;

        auto [wcm, wcm_med] = run_trials([&]() {
            std::fill(B_cm.begin(), B_cm.end(), 0.0);
            rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                               1.0, S, 0, 0, A_cm.data(), m, 0.0, B_cm.data(), d);
        }, num_trials);
        auto [wrm, wrm_med] = run_trials([&]() {
            std::fill(B_rm.begin(), B_rm.end(), 0.0);
            rb::sketch_general(Layout::RowMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                               1.0, S, 0, 0, A_rm.data(), n, 0.0, B_rm.data(), n);
        }, num_trials);

        // COLD ColMajor: fresh unsampled operator each trial (sample+convert+kernel).
        auto [ccm, ccm_med] = run_trials([&]() {
            SparseSkOp<T> Sc(dist, st);
            std::fill(B_cm.begin(), B_cm.end(), 0.0);
            rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                               1.0, Sc, 0, 0, A_cm.data(), m, 0.0, B_cm.data(), d);
        }, num_trials);
        (void)ccm_med;
        // CONVERT-only: worst-case COO->CSC re-sort (deepcopy + sort).
        auto [cvt, cvt_med] = run_trials([&]() {
            auto cp = view.deepcopy();
            cp.sort_arrays(rb::sparse_data::NonzeroSort::CSC);
        }, num_trials);
        (void)cvt_med;

        // Correctness vs the ColMajor oracle (both layouts).
        std::fill(result.begin(), result.end(), 0.0);
        rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                           1.0, S, 0, 0, A_cm.data(), m, 0.0, result.data(), d);
        double maxdiff = 0;
        for (int64_t i = 0; i < d * n; ++i)
            maxdiff = std::max(maxdiff, std::abs(result[i] - B_ref[i]));
        if (maxdiff > 1e-9)
            std::cout << "  FAIL " << spec.label << " ColMajor max|diff|="
                      << std::scientific << maxdiff << std::fixed << "\n";
        std::fill(result.begin(), result.end(), 0.0);
        rb::sketch_general(Layout::RowMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                           1.0, S, 0, 0, A_rm.data(), n, 0.0, result.data(), n);
        maxdiff = 0;
        for (int64_t i = 0; i < d; ++i)
            for (int64_t j = 0; j < n; ++j)
                maxdiff = std::max(maxdiff, std::abs(result[i * n + j] - B_ref[i + j * d]));
        if (maxdiff > 1e-9)
            std::cout << "  FAIL " << spec.label << " RowMajor max|diff|="
                      << std::scientific << maxdiff << std::fixed << "\n";

        // A-read traffic is bounded by the DISTINCT rows of A touched (<= min(nnz,m)),
        // not all m rows: a wide LASO has mostly-empty columns. Using m here would
        // overcount sparse operators (and push %STR past 100%).
        int64_t a_read = std::min(nnz, m) * n;
        Row rc; rc.label = spec.label; rc.min_us = wcm; rc.med_us = wcm_med;
        fill_metrics(rc, wcm, nnz, n, a_read, d * n);
        rc.notes = base_note + " cold=" + std::to_string(ccm) + " cvt=" + std::to_string(cvt);
        rows_cm.push_back(rc);

        Row rr; rr.label = spec.label; rr.min_us = wrm; rr.med_us = wrm_med;
        fill_metrics(rr, wrm, nnz, n, a_read, d * n);
        rr.notes = base_note;
        rows_rm.push_back(rr);
    }

    print_metric_header("warm apply, ColMajor dense  (jik/jki kernels)");
    for (auto& r : rows_cm) print_metric_row(r);
    print_metric_row(oracle);
    std::cout << "\n";
    print_metric_header("warm apply, RowMajor dense  (ikb/kib kernels)");
    for (auto& r : rows_rm) print_metric_row(r);
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// RIGHT-SKETCH:  B(m x d) = A(m x n) * S(n x d),  S ~ SparseDist(n, d, k, axis).
// work_mult = m; read_dense(A) = m*n; out_elems(B) = m*d.
// ---------------------------------------------------------------------------
void run_right(int64_t d, int64_t m, int64_t n,
               const std::vector<OpSpec>& specs, int num_trials) {
    using T = double;
    namespace rb = RandBLAS;
    uint64_t seed = 67890;

    std::cout << "=== RIGHT-SKETCH  B(" << m << "x" << d << ") = A(" << m << "x" << n
              << ") * S(" << n << "x" << d << "),  trials=" << num_trials << " ===\n\n";

    std::vector<T> A_cm(m * n), A_rm(m * n);
    rb::DenseDist DA(m, n);
    auto st = rb::fill_dense(DA, A_cm.data(), rb::RNGState<>(seed));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            A_rm[i * n + j] = A_cm[i + j * m];

    std::vector<T> B_cm(m * d), B_rm(m * d), B_ref(m * d), result(m * d);
    std::vector<T> S_dense(n * d);

    std::vector<Row> rows_cm, rows_rm;

    for (const auto& spec : specs) {
        SparseDist dist(n, d, spec.vec_nnz, spec.axis);
        SparseSkOp<T> S(dist, st);
        rb::fill_sparse(S);
        auto view = rb::sparse::coo_view_of_skop(S);
        int64_t nnz = S.nnz;
        std::string base_note = std::string(sort_name(view.sort)) + " nnz=" + std::to_string(nnz);

        rb::sparse_data::coo::coo_to_dense(view, Layout::ColMajor, S_dense.data());
        std::fill(B_ref.begin(), B_ref.end(), 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, d, n,
                   1.0, A_cm.data(), m, S_dense.data(), n, 0.0, B_ref.data(), m);

        auto [wcm, wcm_med] = run_trials([&]() {
            std::fill(B_cm.begin(), B_cm.end(), 0.0);
            rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, d, n,
                               1.0, A_cm.data(), m, S, 0.0, B_cm.data(), m);
        }, num_trials);
        auto [wrm, wrm_med] = run_trials([&]() {
            std::fill(B_rm.begin(), B_rm.end(), 0.0);
            rb::sketch_general(Layout::RowMajor, Op::NoTrans, Op::NoTrans, m, d, n,
                               1.0, A_rm.data(), n, S, 0.0, B_rm.data(), d);
        }, num_trials);

        std::fill(result.begin(), result.end(), 0.0);
        rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, d, n,
                           1.0, A_cm.data(), m, S, 0.0, result.data(), m);
        double maxdiff = 0;
        for (int64_t i = 0; i < m * d; ++i)
            maxdiff = std::max(maxdiff, std::abs(result[i] - B_ref[i]));
        if (maxdiff > 1e-9)
            std::cout << "  FAIL " << spec.label << " ColMajor max|diff|="
                      << std::scientific << maxdiff << std::fixed << "\n";

        // A-read traffic bounded by distinct columns of A touched (<= min(nnz,n)).
        int64_t a_read = std::min(nnz, n) * m;
        Row rc; rc.label = spec.label; rc.min_us = wcm; rc.med_us = wcm_med;
        fill_metrics(rc, wcm, nnz, m, a_read, m * d); rc.notes = base_note;
        rows_cm.push_back(rc);
        Row rr; rr.label = spec.label; rr.min_us = wrm; rr.med_us = wrm_med;
        fill_metrics(rr, wrm, nnz, m, a_read, m * d); rr.notes = base_note;
        rows_rm.push_back(rr);
    }

    print_metric_header("warm apply, ColMajor dense");
    for (auto& r : rows_cm) print_metric_row(r);
    std::cout << "\n";
    print_metric_header("warm apply, RowMajor dense");
    for (auto& r : rows_rm) print_metric_row(r);
    std::cout << "\n";
}

static bool run_sampling_scaling(
    const OpSpec &spec,
    const SparseDist &dist,
    const RandBLAS::RNGState<> &seed_state,
    int num_trials,
    const std::vector<int> &threads
) {
    const int64_t capacity = dist.full_nnz;
    std::vector<double> values(capacity);
    std::vector<int64_t> rows(capacity);
    std::vector<int64_t> cols(capacity);
    std::vector<double> expected_values;
    std::vector<int64_t> expected_rows;
    std::vector<int64_t> expected_cols;
    auto expected_state = seed_state;
    int64_t expected_nnz = -1;
    std::vector<SamplingScalingRow> scaling_rows;
    scaling_rows.reserve(threads.size());
    long baseline_us = 0;
    int baseline_threads = 1;
    bool exact = true;

    for (int thread_count : threads) {
        set_threads(thread_count);
        const bool uses_permutation_workspace =
            dist.major_axis == Axis::Short && dist.vec_nnz > 1;
        const int policy_threads = RandBLAS::sparse::sparse_sampling_thread_count(
            dist.dim_major,
            dist.dim_minor,
            dist.vec_nnz,
            uses_permutation_workspace
        );
        const int actual_threads = effective_threads(policy_threads);
        int64_t sampled_nnz = -1;
        auto end_state = RandBLAS::fill_sparse_unpacked(
            dist,
            dist.n_rows,
            dist.n_cols,
            0,
            0,
            sampled_nnz,
            values.data(),
            rows.data(),
            cols.data(),
            seed_state
        );
        if (scaling_rows.empty()) {
            expected_nnz = sampled_nnz;
            expected_values = values;
            expected_rows = rows;
            expected_cols = cols;
            expected_state = end_state;
        } else {
            exact = exact
                && sampled_nnz == expected_nnz
                && std::equal(
                    values.begin(), values.begin() + sampled_nnz,
                    expected_values.begin()
                )
                && std::equal(
                    rows.begin(), rows.begin() + sampled_nnz,
                    expected_rows.begin()
                )
                && std::equal(
                    cols.begin(), cols.begin() + sampled_nnz,
                    expected_cols.begin()
                )
                && end_state == expected_state;
        }

        auto [min_us, median_us] = run_trials([&]() {
            sampled_nnz = -1;
            end_state = RandBLAS::fill_sparse_unpacked(
                dist,
                dist.n_rows,
                dist.n_cols,
                0,
                0,
                sampled_nnz,
                values.data(),
                rows.data(),
                cols.data(),
                seed_state
            );
        }, num_trials);
        if (scaling_rows.empty()) {
            baseline_us = min_us;
            baseline_threads = actual_threads;
        }
        const double speedup = min_us > 0
            ? static_cast<double>(baseline_us) / static_cast<double>(min_us)
            : -1.0;
        const double relative_threads = static_cast<double>(actual_threads)
            / static_cast<double>(baseline_threads);
        scaling_rows.push_back({
            thread_count,
            actual_threads,
            min_us,
            median_us,
            static_cast<double>(min_us) * 1000.0
                / static_cast<double>(sampled_nnz),
            speedup,
            speedup / relative_threads
        });
    }

    std::cout << "  " << spec.label << " sampling  (nnz=" << expected_nnz << ")\n"
              << "  " << std::right << std::setw(6) << "Req"
              << std::setw(6) << "Thr"
              << std::setw(10) << "Min(us)"
              << std::setw(10) << "Med(us)"
              << std::setw(13) << "ns/nonzero"
              << std::setw(10) << "Spd(min)"
              << std::setw(10) << "Eff(min)" << "\n"
              << "  " << std::string(65, '-') << "\n";
    for (const SamplingScalingRow &row : scaling_rows) {
        std::cout << "  " << std::right << std::setw(6) << row.requested_threads
                  << std::setw(6) << row.threads
                  << std::setw(10) << row.min_us
                  << std::setw(10) << row.median_us
                  << std::setw(13) << fcell(row.ns_per_nonzero, 2)
                  << std::setw(10) << fcell(row.speedup, 2)
                  << std::setw(10) << fcell(row.efficiency, 2) << "\n";
    }
    std::cout << "  exact output/state: " << (exact ? "PASS" : "FAIL") << "\n\n";
    return exact;
}

// ---------------------------------------------------------------------------
// SCALING: re-run sparse sampling and the left-sketch ColMajor warm apply at
// each thread count. STREAM is recalibrated per thread count for the application
// table, so %STR is apples-to-apples. Note: a memory-bound kernel that has
// already saturated bandwidth SHOULD show efficiency < 1 -- read efficiency
// next to %STR, not in isolation.
// ---------------------------------------------------------------------------
bool run_scaling(int64_t d, int64_t m, int64_t n, const std::vector<OpSpec>& specs,
                 int num_trials, const std::vector<int>& threads, bool do_stream) {
    using T = double;
    namespace rb = RandBLAS;
    uint64_t seed = 12345;

    std::cout << "=== SCALING (sampling + left-sketch, ColMajor warm)  d="
              << d << " m=" << m
              << " n=" << n << ",  trials=" << num_trials << " ===\n";
#if !defined(RandBLAS_HAS_OpenMP)
    std::cout << "  (built without OpenMP -- thread sweep is a no-op)\n";
#endif
    std::cout << "\n";

    std::vector<T> A_cm(m * n);
    rb::DenseDist DA(m, n);
    auto st = rb::fill_dense(DA, A_cm.data(), rb::RNGState<>(seed));
    std::vector<T> B_cm(d * n);
    bool sampling_exact = true;

    for (const auto& spec : specs) {
        SparseDist dist(d, m, spec.vec_nnz, spec.axis);
        sampling_exact = run_sampling_scaling(
            spec, dist, st, num_trials, threads
        ) && sampling_exact;
        SparseSkOp<T> S(dist, st);
        rb::fill_sparse(S);
        int64_t nnz = S.nnz;

        std::cout << "  " << spec.label << "  (nnz=" << nnz << ")\n";
        std::cout << "  " << std::right << std::setw(6) << "Req"
                  << std::setw(6) << "Thr"
                  << std::setw(10) << "Min(us)" << std::setw(10) << "Spd(min)"
                  << std::setw(10) << "Eff(min)" << std::setw(9) << "GB/s"
                  << std::setw(7) << "%STR" << "\n";
        std::cout << "  " << std::string(56, '-') << "\n";

        long base_us = 0;
        int base_threads = 1;
        bool first_thread_count = true;
        for (int t : threads) {
            set_threads(t);
            const int actual_threads = effective_threads(t);
            double stream = do_stream ? measure_stream_gbps() : -1.0;
            auto [mn, md] = run_trials([&]() {
                std::fill(B_cm.begin(), B_cm.end(), 0.0);
                rb::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                                   1.0, S, 0, 0, A_cm.data(), m, 0.0, B_cm.data(), d);
            }, num_trials);
            (void)md;
            if (first_thread_count) {
                base_us = mn;
                base_threads = actual_threads;
                first_thread_count = false;
            }
            double speedup = (mn > 0) ? (double)base_us / (double)mn : 0.0;
            double eff = speedup
                / ((double)actual_threads / (double)base_threads);
            double bytes = (double)(std::min(nnz, m) * n + 2 * d * n) * sizeof(double)
                         + (double)nnz * (sizeof(double) + sizeof(int64_t));
            double gbps = (mn > 0) ? bytes / (double)mn / 1000.0 : -1;
            double pct = (stream > 0 && gbps > 0) ? gbps / stream * 100.0 : -1;
            std::cout << "  " << std::right << std::setw(6) << t
                      << std::setw(6) << actual_threads
                      << std::setw(10) << mn
                      << std::setw(10) << fcell(speedup, 2)
                      << std::setw(10) << fcell(eff, 2)
                      << std::setw(9) << fcell(gbps, 1)
                      << std::setw(7) << fcell(pct, 0) << "\n";
        }
        std::cout << "\n";
    }
    return sampling_exact;
}

// ---------------------------------------------------------------------------
// CSR-vs-CSC KERNEL PROBE (left-sketch). Builds the operator's CSC and CSR
// representations once and times left_spmm DIRECTLY on each (no per-call COO
// re-sort), in both layouts, to isolate the kernel. The question: does the CSR
// jik path -- whose ColMajor outer loop runs over d rows (all non-empty for a
// wide LASO) -- recover the gap vs the CSC jki path, whose outer loop runs over
// m mostly-empty columns? CSC kib (RowMajor) is the already-fast baseline.
//
// Finding: CSR jik (ColMajor) beats CSC jki (ColMajor) 1.1-2x for BOTH SASO and
// LASO with no regression, motivating the ColMajor "prefer CSR for wide
// operators" routing now in apply_coo_via_csx (coo_spmm_impl.hh). (A strided-axpy
// "ColMajor kib" kernel was also tried and rejected -- it regressed dense-column
// SASO ~3x.) The probe times the kernels directly, so it is unaffected by that
// routing change and remains the A/B harness for it.
// ---------------------------------------------------------------------------
void run_csr_probe(int64_t d, int64_t m, int64_t n,
                   const std::vector<OpSpec>& specs, int num_trials) {
    using T = double;
    namespace rb = RandBLAS;
    uint64_t seed = 12345;

    std::cout << "=== CSR-vs-CSC KERNEL PROBE (left-sketch)  d=" << d << " m=" << m
              << " n=" << n << ",  trials=" << num_trials << " ===\n";
    std::cout << "  kernels timed directly via left_spmm on prebuilt CSC/CSR (no COO re-sort)\n\n";

    std::vector<T> A_cm(m * n), A_rm(m * n);
    rb::DenseDist DA(m, n);
    auto st = rb::fill_dense(DA, A_cm.data(), rb::RNGState<>(seed));
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            A_rm[i * n + j] = A_cm[i + j * m];
    std::vector<T> C_cm(d * n), C_rm(d * n), B_ref(d * n), S_dense(d * m), result(d * n);

    for (const auto& spec : specs) {
        SparseDist dist(d, m, spec.vec_nnz, spec.axis);
        SparseSkOp<T> S(dist, st);
        rb::fill_sparse(S);
        auto view = rb::sparse::coo_view_of_skop(S);
        int64_t nnz = S.nnz;
        auto S_csc = view.as_owning_csc();
        auto S_csr = view.as_owning_csr();
        int64_t a_read = std::min(nnz, m) * n;

        // Oracle: densify(S) + GEMM (also the correctness reference).
        rb::sparse_data::coo::coo_to_dense(view, Layout::ColMajor, S_dense.data());
        std::fill(B_ref.begin(), B_ref.end(), 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                   1.0, S_dense.data(), d, A_cm.data(), m, 0.0, B_ref.data(), d);

        std::vector<Row> rows;
        auto probe = [&](const std::string& label, const auto& Sp, Layout lay) {
            const T* Bp = (lay == Layout::ColMajor) ? A_cm.data() : A_rm.data();
            int64_t ldb = (lay == Layout::ColMajor) ? m : n;
            T*      Cp  = (lay == Layout::ColMajor) ? C_cm.data() : C_rm.data();
            int64_t ldc = (lay == Layout::ColMajor) ? d : n;
            auto [mn, md] = run_trials([&]() {
                std::fill(Cp, Cp + d * n, 0.0);
                rb::sparse_data::left_spmm(lay, Op::NoTrans, Op::NoTrans, d, n, m,
                                           1.0, Sp, 0, 0, Bp, ldb, 0.0, Cp, ldc);
            }, num_trials);
            std::fill(result.begin(), result.end(), 0.0);
            rb::sparse_data::left_spmm(lay, Op::NoTrans, Op::NoTrans, d, n, m,
                                       1.0, Sp, 0, 0, Bp, ldb, 0.0, result.data(), ldc);
            double maxdiff = 0;
            if (lay == Layout::ColMajor)
                for (int64_t i = 0; i < d * n; ++i)
                    maxdiff = std::max(maxdiff, std::abs(result[i] - B_ref[i]));
            else
                for (int64_t i = 0; i < d; ++i)
                    for (int64_t j = 0; j < n; ++j)
                        maxdiff = std::max(maxdiff, std::abs(result[i * n + j] - B_ref[i + j * d]));
            if (maxdiff > 1e-9)
                std::cout << "  FAIL " << label << " max|diff|=" << std::scientific
                          << maxdiff << std::fixed << "\n";
            Row r; r.label = label; r.min_us = mn; r.med_us = md;
            fill_metrics(r, mn, nnz, n, a_read, d * n);
            r.notes = std::string(sort_name(view.sort)) + " nnz=" + std::to_string(nnz);
            rows.push_back(r);
        };

        print_metric_header(spec.label);
        probe("CSC jki  ColMajor", S_csc, Layout::ColMajor);
        probe("CSR jik  ColMajor", S_csr, Layout::ColMajor);
        probe("CSC kib  RowMajor", S_csc, Layout::RowMajor);
        probe("CSR ikb  RowMajor", S_csr, Layout::RowMajor);
        for (auto& r : rows) print_metric_row(r);
        std::cout << "\n";
    }
}

std::vector<OpSpec> sweep_specs() {
    return {
        {"Wide SASO  k=2",  2, Axis::Short},
        {"Wide SASO  k=4",  4, Axis::Short},
        {"Wide SASO  k=8",  8, Axis::Short},
        {"CountSketch k=1", 1, Axis::Short},
        {"Wide LASO  k=2",  2, Axis::Long},
        {"Wide LASO  k=4",  4, Axis::Long},
        {"Wide LASO  k=8",  8, Axis::Long},
    };
}

static std::vector<int> parse_threads(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    }
    if (out.empty()) out = {1, 2, 4, 8};
    return out;
}

static bool thread_counts_are_valid(const std::vector<int> &thread_counts) {
    return std::all_of(
        thread_counts.begin(),
        thread_counts.end(),
        [](int thread_count) { return thread_count > 0; }
    );
}

int main(int argc, char** argv) {
    OpenMPSettingsGuard openmp_settings;
    bool no_stream = false, scaling = false, csr_probe = false;
    std::vector<int> threads = {1, 2, 4, 8};
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--no-stream")      no_stream = true;
        else if (s == "--scaling")   scaling = true;
        else if (s == "--csr-probe") csr_probe = true;
        else if (s.rfind("--threads=", 0) == 0) threads = parse_threads(s.substr(10));
        else pos.push_back(s);
    }
    if (!thread_counts_are_valid(threads)) {
        std::cerr << "Invalid thread list. Expected positive integers.\n";
        return 1;
    }

    std::cout << "\n============================================================\n";
    std::cout << "SKETCH_GENERAL PERFORMANCE BENCHMARK (sparse operators)\n";
    std::cout << "============================================================\n";
    std::cout << "Threads (OMP_NUM_THREADS/max): " << current_threads() << "\n";

    bool have_cfg = pos.size() >= 5;
    int64_t d = have_cfg ? std::atoll(pos[0].c_str()) : 0;
    int64_t m = have_cfg ? std::atoll(pos[1].c_str()) : 0;
    int64_t n = have_cfg ? std::atoll(pos[2].c_str()) : 0;
    int64_t k = have_cfg ? std::atoll(pos[3].c_str()) : 0;
    Axis axis = (have_cfg && std::atoi(pos[4].c_str()) != 0) ? Axis::Long : Axis::Short;
    int trials = (pos.size() > 5) ? std::atoi(pos[5].c_str()) : (have_cfg ? 20 : 10);

    if (scaling) {
        // STREAM is calibrated per thread count inside run_scaling.
        std::vector<OpSpec> specs = have_cfg
            ? std::vector<OpSpec>{{"single", k, axis}} : sweep_specs();
        int64_t sd = have_cfg ? d : 200, sm = have_cfg ? m : 2000, sn = have_cfg ? n : 2000;
        std::cout << "\n";
        return run_scaling(
            sd, sm, sn, specs, trials, threads, !no_stream
        ) ? 0 : 2;
    }

    if (!no_stream) {
        std::cout << "Calibrating STREAM triad..." << std::flush;
        g_stream_gbps = measure_stream_gbps();
        std::cout << " " << std::fixed << std::setprecision(1) << g_stream_gbps
                  << " GB/s  (= 100% on the %STR column)\n";
    } else {
        std::cout << "STREAM calibration skipped (--no-stream); %STR shows '-'\n";
    }
    std::cout << "\n";

    if (csr_probe) {
        std::vector<OpSpec> specs = have_cfg
            ? std::vector<OpSpec>{{"single", k, axis}} : sweep_specs();
        int64_t sd = have_cfg ? d : 200, sm = have_cfg ? m : 2000, sn = have_cfg ? n : 2000;
        run_csr_probe(sd, sm, sn, specs, trials);
        return 0;
    }

    if (have_cfg) {
        std::vector<OpSpec> specs = {{"single", k, axis}};
        run_left(d, m, n, specs, trials);
        run_right(d, m, n, specs, trials);
        return 0;
    }

    auto specs = sweep_specs();
    std::cout << "########## SECTION 1: vary sketch size d (m=n=2000) ##########\n\n";
    for (int64_t dd : {40, 200, 500}) {
        run_left(dd, 2000, 2000, specs, trials);
        run_right(dd, 2000, 2000, specs, trials);
    }
    std::cout << "########## SECTION 2: narrow n (SpMV-ish) ##########\n\n";
    run_left(200, 2000, 1, specs, trials);
    run_left(200, 2000, 10, specs, trials);
    return 0;
}
