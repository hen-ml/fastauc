//! Native core of `fastauc`.
//!
//! Everything here assumes inputs were normalised by the Python wrapper
//! (`python/fastauc/__init__.py`): labels are `u8` (non-zero = positive),
//! scores and weights are `f64`. Shapes and value ranges are still checked
//! here, so calling `fastauc._core` directly cannot read out of bounds.
//!
//! There is no `unsafe` in this file.

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

/// Below this length a sort is done on the calling thread.
const PAR_SORT_MIN: usize = 1 << 16;

type ArrPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

/// Optional dedicated pool set by `set_num_threads`; `None` = rayon's global pool.
static POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

/// Run `f` with the GIL released, inside the configured thread pool.
fn detached<R: Send>(py: Python<'_>, f: impl FnOnce() -> R + Send) -> R {
    let pool = POOL.read().ok().and_then(|p| p.clone());
    py.detach(move || match pool {
        Some(pool) => pool.install(f),
        None => f(),
    })
}

/// Limit fastauc to `n` threads; 0 restores the default (all cores, or
/// `RAYON_NUM_THREADS`).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    let pool = if n == 0 {
        None
    } else {
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("fastauc-{i}"))
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Some(Arc::new(built))
    };
    *POOL.write().map_err(|_| PyValueError::new_err("thread pool lock poisoned"))? = pool;
    Ok(())
}

/// Number of worker threads fastauc will use.
#[pyfunction]
fn num_threads() -> usize {
    match POOL.read().ok().and_then(|p| p.clone()) {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// Map a non-NaN f64 to a u64 whose ascending order is the *descending*
/// order of the floats. `-0.0` and `+0.0` map to the same key.
#[inline(always)]
fn desc_key(x: f64) -> u64 {
    let b = (x + 0.0).to_bits(); // -0.0 + 0.0 == +0.0
    let asc = if b >> 63 == 0 { b ^ (1 << 63) } else { !b };
    !asc
}

/// Descending order of a score vector plus the ends of its tied blocks.
struct Sorted {
    /// `order[i]` is the original index of the i-th highest score.
    order: Vec<u32>,
    /// `tie_end[i]` is true when position `i` is the last of a run of equal scores.
    tie_end: Vec<bool>,
}

fn sort_desc<I: Iterator<Item = f64>>(scores: I) -> Sorted {
    let mut keyed: Vec<(u64, u32)> = scores
        .enumerate()
        .map(|(i, s)| (desc_key(s), i as u32))
        .collect();
    // Integer keys sort much faster than an indirect float comparison, and
    // including the index makes the order fully deterministic.
    if keyed.len() >= PAR_SORT_MIN {
        keyed.par_sort_unstable();
    } else {
        keyed.sort_unstable();
    }
    let n = keyed.len();
    let tie_end = (0..n)
        .map(|i| i + 1 == n || keyed[i].0 != keyed[i + 1].0)
        .collect();
    let order = keyed.into_iter().map(|(_, i)| i).collect();
    Sorted { order, tie_end }
}

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

/// One pass over a descending order, producing (ROC-AUC, average precision).
///
/// `is_pos` and `weight` are looked up by *original* sample index. Tied scores
/// are treated as a single threshold, which matches scikit-learn exactly:
/// trapezoidal ROC area, and step-wise (non-interpolated) average precision.
///
/// ROC-AUC is NaN unless both classes have positive total weight.
/// Average precision is NaN when there is no positive weight.
#[inline(always)]
fn scan<L, W>(s: &Sorted, is_pos: L, weight: W) -> (f64, f64)
where
    L: Fn(usize) -> bool,
    W: Fn(usize) -> f64,
{
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    let (mut prev_tp, mut prev_fp) = (0.0f64, 0.0f64);
    let (mut roc, mut ap) = (0.0f64, 0.0f64);

    for (&idx, &end) in s.order.iter().zip(s.tie_end.iter()) {
        let idx = idx as usize;
        let w = weight(idx);
        if is_pos(idx) {
            tp += w;
        } else {
            fp += w;
        }
        if end {
            roc += (fp - prev_fp) * (tp + prev_tp) * 0.5;
            if tp > prev_tp {
                ap += (tp - prev_tp) * (tp / (tp + fp));
            }
            prev_tp = tp;
            prev_fp = fp;
        }
    }

    let roc = if tp > 0.0 && fp > 0.0 { roc / (tp * fp) } else { f64::NAN };
    let ap = if tp > 0.0 { ap / tp } else { f64::NAN };
    (roc, ap)
}

#[inline(always)]
fn scan_opt<L: Fn(usize) -> bool>(s: &Sorted, is_pos: L, w: Option<&[f64]>) -> (f64, f64) {
    match w {
        Some(w) => scan(s, is_pos, |i| w[i]),
        None => scan(s, is_pos, |_| 1.0),
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn check_len(n: usize) -> PyResult<()> {
    if n > u32::MAX as usize {
        return err("fastauc supports at most 2**32 - 1 samples per vector");
    }
    Ok(())
}

fn check_same(what: &str, a: usize, b: usize) -> PyResult<()> {
    if a != b {
        return err(format!("{what}: length mismatch ({a} vs {b})"));
    }
    Ok(())
}

fn check_scores<'a, I: IntoIterator<Item = &'a f64>>(it: I) -> PyResult<()> {
    if it.into_iter().any(|x| x.is_nan()) {
        return err("y_score contains NaN");
    }
    Ok(())
}

fn check_weights(w: Option<&[f64]>, n: usize) -> PyResult<()> {
    if let Some(w) = w {
        check_same("sample_weight", w.len(), n)?;
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return err("sample_weight must be finite and non-negative");
        }
    }
    Ok(())
}

fn weights_slice<'a>(w: &'a Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<&'a [f64]>> {
    match w {
        Some(w) => Ok(Some(w.as_slice()?)),
        None => Ok(None),
    }
}

fn to_py<'py>(py: Python<'py>, pair: (Vec<f64>, Vec<f64>)) -> ArrPair<'py> {
    (pair.0.into_pyarray(py), pair.1.into_pyarray(py))
}

#[inline(always)]
fn row_pos<'a>(row: ArrayView1<'a, u8>) -> impl Fn(usize) -> bool + 'a {
    move |i| row[i] != 0
}

// ---------------------------------------------------------------------------
// Single pair
// ---------------------------------------------------------------------------

/// (roc_auc, average_precision) for one label vector and one score vector.
#[pyfunction]
#[pyo3(signature = (y_true, y_score, sample_weight=None))]
fn roc_auc_ap(
    py: Python<'_>,
    y_true: PyReadonlyArray1<'_, u8>,
    y_score: PyReadonlyArray1<'_, f64>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(f64, f64)> {
    let yt = y_true.as_slice()?;
    let ys = y_score.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_len(ys.len())?;
    check_same("y_true / y_score", yt.len(), ys.len())?;
    check_weights(w, ys.len())?;
    check_scores(ys)?;

    Ok(detached(py, || {
        let sorted = sort_desc(ys.iter().copied());
        scan_opt(&sorted, |i| yt[i] != 0, w)
    }))
}

// ---------------------------------------------------------------------------
// Batched
// ---------------------------------------------------------------------------

/// M score rows (M, N) against one label vector (N,). One sort per row, in parallel.
#[pyfunction]
#[pyo3(signature = (y_true, y_scores, sample_weight=None))]
fn many_scores<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_scores: PyReadonlyArray2<'py, f64>,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<ArrPair<'py>> {
    let yt = y_true.as_slice()?;
    let ys: ArrayView2<f64> = y_scores.as_array();
    let w = weights_slice(&sample_weight)?;
    check_len(ys.ncols())?;
    check_same("y_true / y_score", yt.len(), ys.ncols())?;
    check_weights(w, yt.len())?;
    check_scores(ys.iter())?;

    let out = detached(py, || {
        (0..ys.nrows())
            .into_par_iter()
            .map(|r| {
                let sorted = sort_desc(ys.row(r).iter().copied());
                scan_opt(&sorted, |i| yt[i] != 0, w)
            })
            .unzip()
    });
    Ok(to_py(py, out))
}

/// M label rows (M, N) against one score vector (N,). Sorts once, then each
/// row is a single O(N) scan.
#[pyfunction]
#[pyo3(signature = (y_trues, y_score, sample_weight=None))]
fn many_targets<'py>(
    py: Python<'py>,
    y_trues: PyReadonlyArray2<'py, u8>,
    y_score: PyReadonlyArray1<'py, f64>,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<ArrPair<'py>> {
    let yt: ArrayView2<u8> = y_trues.as_array();
    let ys = y_score.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_len(ys.len())?;
    check_same("y_true / y_score", yt.ncols(), ys.len())?;
    check_weights(w, ys.len())?;
    check_scores(ys)?;

    let out = detached(py, || {
        let sorted = sort_desc(ys.iter().copied());
        (0..yt.nrows())
            .into_par_iter()
            .map(|r| scan_opt(&sorted, row_pos(yt.row(r)), w))
            .unzip()
    });
    Ok(to_py(py, out))
}

/// Row-wise pairs: label row i (M, N) against score row i (M, N).
#[pyfunction]
#[pyo3(signature = (y_trues, y_scores, sample_weight=None))]
fn pairwise<'py>(
    py: Python<'py>,
    y_trues: PyReadonlyArray2<'py, u8>,
    y_scores: PyReadonlyArray2<'py, f64>,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<ArrPair<'py>> {
    let yt: ArrayView2<u8> = y_trues.as_array();
    let ys: ArrayView2<f64> = y_scores.as_array();
    let w = weights_slice(&sample_weight)?;
    check_len(ys.ncols())?;
    check_same("y_true / y_score rows", yt.nrows(), ys.nrows())?;
    check_same("y_true / y_score", yt.ncols(), ys.ncols())?;
    check_weights(w, ys.ncols())?;
    check_scores(ys.iter())?;

    let out = detached(py, || {
        (0..ys.nrows())
            .into_par_iter()
            .map(|r| {
                let sorted = sort_desc(ys.row(r).iter().copied());
                scan_opt(&sorted, row_pos(yt.row(r)), w)
            })
            .unzip()
    });
    Ok(to_py(py, out))
}

// ---------------------------------------------------------------------------
// Grouped (per-query / per-user AUC)
// ---------------------------------------------------------------------------

/// Per-group metrics. `codes[i]` in `0..n_groups` is the group of sample i.
/// Returns (roc_auc, average_precision), each of length `n_groups`.
#[pyfunction]
fn grouped<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_score: PyReadonlyArray1<'py, f64>,
    codes: PyReadonlyArray1<'py, i64>,
    n_groups: usize,
) -> PyResult<ArrPair<'py>> {
    let yt = y_true.as_slice()?;
    let ys = y_score.as_slice()?;
    let codes = codes.as_slice()?;
    let n = ys.len();
    check_len(n)?;
    check_same("y_true / y_score", yt.len(), n)?;
    check_same("groups / y_score", codes.len(), n)?;
    check_scores(ys)?;
    if codes.iter().any(|&c| c < 0 || c as u64 >= n_groups as u64) {
        return err("group codes must lie in [0, n_groups)");
    }

    let out = detached(py, || {
        // Counting sort of sample indices by group.
        let mut start = vec![0usize; n_groups + 1];
        for &c in codes {
            start[c as usize + 1] += 1;
        }
        for g in 0..n_groups {
            start[g + 1] += start[g];
        }
        let mut cursor = start.clone();
        let mut members = vec![0u32; n];
        for (i, &c) in codes.iter().enumerate() {
            let slot = &mut cursor[c as usize];
            members[*slot] = i as u32;
            *slot += 1;
        }

        (0..n_groups)
            .into_par_iter()
            .map(|g| {
                let m = &members[start[g]..start[g + 1]];
                let sorted = sort_desc(m.iter().map(|&i| ys[i as usize]));
                scan(&sorted, |j| yt[m[j] as usize] != 0, |_| 1.0)
            })
            .unzip()
    });
    Ok(to_py(py, out))
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

#[inline(always)]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++, seeded through SplitMix64 from (seed, replicate).
///
/// Every replicate gets its own independently mixed state, so replicates are
/// statistically independent and the output does not depend on thread count.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64, replicate: u64) -> Self {
        const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut sm = mix64(seed.wrapping_add(GOLDEN))
            ^ mix64(replicate.wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(GOLDEN));
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            sm = sm.wrapping_add(GOLDEN);
            *x = mix64(sm);
        }
        if s == [0; 4] {
            s[0] = GOLDEN;
        }
        Rng(s)
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform integer in `0..n` (multiply-shift; bias < 2^-32 for n < 2^32).
    #[inline(always)]
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// Bootstrap K score rows (K, N) against one label vector.
///
/// Each replicate draws one resample (as multiplicities) and evaluates all K
/// models on it, so replicates are paired across models. Scores are sorted
/// once per model; each replicate is O(N) per model.
///
/// With `codes`, whole groups are resampled (cluster bootstrap): sample `i`
/// gets the multiplicity drawn for group `codes[i]`.
///
/// Returns two flat arrays of length `n_bootstraps * K`, replicate-major.
#[pyfunction]
#[pyo3(signature = (y_true, y_scores, n_bootstraps, seed, stratified, sample_weight=None, codes=None, n_groups=0))]
#[allow(clippy::too_many_arguments)]
fn bootstrap<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_scores: PyReadonlyArray2<'py, f64>,
    n_bootstraps: usize,
    seed: u64,
    stratified: bool,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
    codes: Option<PyReadonlyArray1<'py, i64>>,
    n_groups: usize,
) -> PyResult<ArrPair<'py>> {
    let yt = y_true.as_slice()?;
    let ys: ArrayView2<f64> = y_scores.as_array();
    let w = weights_slice(&sample_weight)?;
    let n = yt.len();
    let k = ys.nrows();
    check_len(n)?;
    check_same("y_true / y_score", n, ys.ncols())?;
    check_weights(w, n)?;
    check_scores(ys.iter())?;
    let codes = match &codes {
        Some(c) => Some(c.as_slice()?),
        None => None,
    };
    if let Some(c) = codes {
        check_same("groups / y_score", c.len(), n)?;
        if stratified {
            return err("stratified resampling cannot be combined with groups");
        }
        if c.iter().any(|&c| c < 0 || c as u64 >= n_groups as u64) {
            return err("group codes must lie in [0, n_groups)");
        }
    }
    // Number of resampling units: groups if given, otherwise samples.
    let units = if codes.is_some() { n_groups } else { n };

    let out = detached(py, || {
        let sorted: Vec<Sorted> = (0..k)
            .into_par_iter()
            .map(|r| sort_desc(ys.row(r).iter().copied()))
            .collect();
        let pos: Vec<u32> = (0..n as u32).filter(|&i| yt[i as usize] != 0).collect();
        let neg: Vec<u32> = (0..n as u32).filter(|&i| yt[i as usize] == 0).collect();

        let per_rep: Vec<Vec<(f64, f64)>> = (0..n_bootstraps)
            .into_par_iter()
            .map_init(
                || vec![0u32; units],
                |counts, b| {
                    counts.fill(0);
                    let mut rng = Rng::new(seed, b as u64);
                    if stratified {
                        for class in [&pos, &neg] {
                            for _ in 0..class.len() {
                                counts[class[rng.below(class.len())] as usize] += 1;
                            }
                        }
                    } else {
                        for _ in 0..units {
                            counts[rng.below(units)] += 1;
                        }
                    }
                    let counts = &*counts;
                    sorted
                        .iter()
                        .map(|s| match (codes, w) {
                            (None, None) => scan(s, |i| yt[i] != 0, |i| counts[i] as f64),
                            (None, Some(w)) => {
                                scan(s, |i| yt[i] != 0, |i| counts[i] as f64 * w[i])
                            }
                            (Some(c), None) => {
                                scan(s, |i| yt[i] != 0, |i| counts[c[i] as usize] as f64)
                            }
                            (Some(c), Some(w)) => {
                                scan(s, |i| yt[i] != 0, |i| counts[c[i] as usize] as f64 * w[i])
                            }
                        })
                        .collect()
                },
            )
            .collect();

        let mut roc = Vec::with_capacity(n_bootstraps * k);
        let mut ap = Vec::with_capacity(n_bootstraps * k);
        for rep in per_rep {
            for (r, a) in rep {
                roc.push(r);
                ap.push(a);
            }
        }
        (roc, ap)
    });
    Ok(to_py(py, out))
}

/// Bootstrap of weighted means over groups.
///
/// `values` is a flat `(n_groups, n_cols)` matrix that may contain NaN
/// (undefined for that group). Each replicate resamples groups with
/// replacement and returns, per column, the `weight`-weighted mean over the
/// drawn groups where the value is defined. Output is flat
/// `(n_bootstraps, n_cols)`.
///
/// The draws depend only on `(seed, replicate, n_groups)`, so separate calls
/// with the same seed are paired.
#[pyfunction]
fn bootstrap_group_means<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    n_cols: usize,
    weight: PyReadonlyArray1<'py, f64>,
    n_bootstraps: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = values.as_slice()?;
    let w = weight.as_slice()?;
    let g = w.len();
    if n_cols == 0 || v.len() != g * n_cols {
        return err("values must have n_groups * n_cols elements");
    }
    check_weights(Some(w), g)?;

    let out: Vec<f64> = detached(py, || {
        let reps: Vec<Vec<f64>> = (0..n_bootstraps)
            .into_par_iter()
            .map_init(
                || vec![0u32; g],
                |counts, b| {
                    counts.fill(0);
                    let mut rng = Rng::new(seed, b as u64);
                    for _ in 0..g {
                        counts[rng.below(g)] += 1;
                    }
                    let mut num = vec![0.0f64; n_cols];
                    let mut den = vec![0.0f64; n_cols];
                    for (gi, &c) in counts.iter().enumerate() {
                        if c == 0 {
                            continue;
                        }
                        let cw = c as f64 * w[gi];
                        for (j, &x) in v[gi * n_cols..(gi + 1) * n_cols].iter().enumerate() {
                            if !x.is_nan() {
                                num[j] += cw * x;
                                den[j] += cw;
                            }
                        }
                    }
                    for j in 0..n_cols {
                        num[j] = if den[j] > 0.0 { num[j] / den[j] } else { f64::NAN };
                    }
                    num
                },
            )
            .collect();
        reps.into_iter().flatten().collect()
    });
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Threshold metrics on the global sort
// ---------------------------------------------------------------------------

fn class_totals(yt: &[u8], w: Option<&[f64]>) -> (f64, f64) {
    let (mut p, mut n) = (0.0, 0.0);
    for (i, &y) in yt.iter().enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        if y != 0 {
            p += wi;
        } else {
            n += wi;
        }
    }
    (p, n)
}

/// (roc_auc, average_precision, ks, ks_threshold, f1_max, f1_threshold, pr_auc)
/// from one sort. "Predict positive when score >= threshold."
///
/// Both PR summaries walk the same curve, one point per distinct score:
/// average precision is the step-wise area sum(dR * P); pr_auc is the
/// trapezoidal area sum(dR * (P + P_prev) / 2) starting from (R, P) = (0, 1),
/// which is what `sklearn.metrics.auc(recall, precision)` computes.
#[pyfunction]
#[pyo3(signature = (y_true, y_score, sample_weight=None))]
fn binary_summary(
    py: Python<'_>,
    y_true: PyReadonlyArray1<'_, u8>,
    y_score: PyReadonlyArray1<'_, f64>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    let yt = y_true.as_slice()?;
    let ys = y_score.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_len(ys.len())?;
    check_same("y_true / y_score", yt.len(), ys.len())?;
    check_weights(w, ys.len())?;
    check_scores(ys)?;

    Ok(detached(py, || {
        let nan = f64::NAN;
        let (mut pr_trap, mut prev_prec) = (0.0f64, 1.0f64);
        let (p_tot, n_tot) = class_totals(yt, w);
        let s = sort_desc(ys.iter().copied());
        let (mut tp, mut fp, mut prev_tp, mut prev_fp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut roc, mut ap) = (0.0f64, 0.0f64);
        let (mut ks, mut ks_thr) = (-1.0f64, nan);
        let (mut f1, mut f1_thr) = (-1.0f64, nan);
        let both = p_tot > 0.0 && n_tot > 0.0;

        for (&idx, &end) in s.order.iter().zip(s.tie_end.iter()) {
            let idx = idx as usize;
            let wi = w.map_or(1.0, |w| w[idx]);
            if yt[idx] != 0 {
                tp += wi;
            } else {
                fp += wi;
            }
            if end {
                roc += (fp - prev_fp) * (tp + prev_tp) * 0.5;
                if tp > prev_tp {
                    ap += (tp - prev_tp) * (tp / (tp + fp));
                }
                if tp + fp > 0.0 {
                    let prec = tp / (tp + fp);
                    pr_trap += (tp - prev_tp) * (prec + prev_prec) * 0.5;
                    prev_prec = prec;
                }
                if both {
                    let d = (tp / p_tot - fp / n_tot).abs();
                    if d > ks {
                        ks = d;
                        ks_thr = ys[idx];
                    }
                }
                if p_tot > 0.0 {
                    let f = 2.0 * tp / (tp + fp + p_tot);
                    if f > f1 {
                        f1 = f;
                        f1_thr = ys[idx];
                    }
                }
                prev_tp = tp;
                prev_fp = fp;
            }
        }
        (
            if both { roc / (p_tot * n_tot) } else { nan },
            if p_tot > 0.0 { ap / p_tot } else { nan },
            if both { ks } else { nan },
            ks_thr,
            if p_tot > 0.0 { f1 } else { nan },
            f1_thr,
            if p_tot > 0.0 { pr_trap / p_tot } else { nan },
        )
    }))
}

/// Cumulative (weighted) true and false positive counts at every distinct
/// score, highest threshold first. Returns (thresholds, tps, fps).
#[pyfunction]
#[pyo3(signature = (y_true, y_score, sample_weight=None))]
#[allow(clippy::type_complexity)]
fn curve<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_score: PyReadonlyArray1<'py, f64>,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let yt = y_true.as_slice()?;
    let ys = y_score.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_len(ys.len())?;
    check_same("y_true / y_score", yt.len(), ys.len())?;
    check_weights(w, ys.len())?;
    check_scores(ys)?;

    let (thr, tps, fps) = detached(py, || {
        let s = sort_desc(ys.iter().copied());
        let (mut thr, mut tps, mut fps) = (Vec::new(), Vec::new(), Vec::new());
        let (mut tp, mut fp) = (0.0f64, 0.0f64);
        for (&idx, &end) in s.order.iter().zip(s.tie_end.iter()) {
            let idx = idx as usize;
            let wi = w.map_or(1.0, |w| w[idx]);
            if yt[idx] != 0 {
                tp += wi;
            } else {
                fp += wi;
            }
            if end {
                thr.push(ys[idx]);
                tps.push(tp);
                fps.push(fp);
            }
        }
        (thr, tps, fps)
    });
    Ok((thr.into_pyarray(py), tps.into_pyarray(py), fps.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Pointwise calibration sums (no sort)
// ---------------------------------------------------------------------------

/// One parallel pass over probabilities in [0, 1]. Returns weighted sums
/// (weight, positives, predictions, log loss, squared error).
#[pyfunction]
#[pyo3(signature = (y_true, y_prob, sample_weight=None, eps=1e-15))]
fn calibration_sums(
    py: Python<'_>,
    y_true: PyReadonlyArray1<'_, u8>,
    y_prob: PyReadonlyArray1<'_, f64>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    eps: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let yt = y_true.as_slice()?;
    let ps = y_prob.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_same("y_true / y_prob", yt.len(), ps.len())?;
    check_weights(w, ps.len())?;
    if ps.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return err("calibration metrics need probabilities in [0, 1]");
    }
    if !(0.0..0.5).contains(&eps) {
        return err("eps must be in [0, 0.5)");
    }

    const CHUNK: usize = 1 << 15;
    Ok(detached(py, || {
        // Fixed chunking + ordered reduction: bit-identical for any thread count.
        let parts: Vec<[f64; 5]> = (0..ps.len().div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| {
                let lo = c * CHUNK;
                let hi = (lo + CHUNK).min(ps.len());
                let mut a = [0.0f64; 5];
                for i in lo..hi {
                    let wi = w.map_or(1.0, |w| w[i]);
                    let y = yt[i] != 0;
                    let p = ps[i];
                    let pc = p.clamp(eps, 1.0 - eps);
                    let (ll, d) = if y { (-pc.ln(), 1.0 - p) } else { (-(1.0 - pc).ln(), p) };
                    a[0] += wi;
                    a[1] += if y { wi } else { 0.0 };
                    a[2] += wi * p;
                    a[3] += wi * ll;
                    a[4] += wi * d * d;
                }
                a
            })
            .collect();
        let mut t = [0.0f64; 5];
        for a in parts {
            for j in 0..5 {
                t[j] += a[j];
            }
        }
        (t[0], t[1], t[2], t[3], t[4])
    }))
}

// ---------------------------------------------------------------------------
// Grouped ranking metrics
// ---------------------------------------------------------------------------

/// Metrics emitted per cutoff, in this order.
const PER_K: usize = 6; // ndcg, mrr, map, recall, precision, hit_rate

struct RankTables {
    /// disc[i] = sum_{r=1..i} 1 / log2(r + 1)
    disc: Vec<f64>,
    /// harm[i] = sum_{r=1..i} 1 / r
    harm: Vec<f64>,
}

impl RankTables {
    fn new(n: usize) -> Self {
        let mut disc = Vec::with_capacity(n + 1);
        let mut harm = Vec::with_capacity(n + 1);
        let (mut d, mut h) = (0.0f64, 0.0f64);
        disc.push(0.0);
        harm.push(0.0);
        for r in 1..=n {
            d += 1.0 / ((r + 1) as f64).log2();
            h += 1.0 / r as f64;
            disc.push(d);
            harm.push(h);
        }
        RankTables { disc, harm }
    }
}

/// Reusable per-thread buffers for `rank_group`, so short lists cost no
/// allocations.
#[derive(Default)]
struct Scratch {
    keyed: Vec<(u64, u32)>,
    tpk: Vec<f64>,
    dcg: Vec<f64>,
    ap: Vec<f64>,
    mrr: Vec<f64>,
    miss: Vec<f64>,
    ideal: Vec<f64>,
}

/// Per-call options shared by every group.
struct RankOpts<'a> {
    ks: &'a [usize],
    /// All gains are 0 or 1: the ideal DCG is a prefix of discounts, no sort.
    binary: bool,
    /// MAP@k divides by R (trec_eval) instead of min(R, k).
    map_by_total: bool,
}

/// Optional complete judgments for one group: the number of judged relevant
/// items and the ideal DCG at each cutoff (computed from all judged gains).
struct Judged<'a> {
    n_relevant: f64,
    idcg: &'a [f64],
}

/// All ranking metrics for one group, written to `out` as
/// `[roc_auc, (ndcg, mrr, map, recall, precision, hit_rate) per cutoff]`.
///
/// Tied scores are handled exactly: every top-k value is its expectation
/// under a uniformly random ordering of the tied items, in closed form.
/// With no ties this is the ordinary definition.
///
/// A cutoff of `usize::MAX` means "the whole list". Without `judged`, the
/// relevant items in the list are taken to be all of them.
#[allow(clippy::too_many_arguments)]
fn rank_group(
    m: &[u32],
    yt: &[u8],
    gains: &[f64],
    ys: &[f64],
    opts: &RankOpts,
    judged: Option<Judged>,
    tab: &RankTables,
    sc: &mut Scratch,
    out: &mut [f64],
) {
    out.fill(f64::NAN);
    let len = m.len();
    if len == 0 {
        return;
    }
    let ks = opts.ks;
    let nk = ks.len();
    // Deepest position any cutoff looks at.
    let kmax = ks.iter().copied().max().unwrap_or(0).min(len);

    sc.keyed.clear();
    sc.keyed.extend(m.iter().map(|&i| (desc_key(ys[i as usize]), i)));
    sc.keyed.sort_unstable();
    let keyed = &sc.keyed;

    for v in [&mut sc.tpk, &mut sc.dcg, &mut sc.ap, &mut sc.mrr] {
        v.clear();
        v.resize(nk, 0.0);
    }
    sc.miss.clear();
    sc.miss.resize(nk, 1.0); // P(no relevant item in the top k)

    let (mut tp, mut fp, mut roc) = (0.0f64, 0.0f64, 0.0f64);
    let (mut s, mut p, mut gsum) = (0usize, 0usize, 0.0f64);

    for i in 0..len {
        let (key, idx) = keyed[i];
        let idx = idx as usize;
        if yt[idx] != 0 {
            p += 1;
        }
        gsum += gains[idx];
        if i + 1 < len && keyed[i + 1].0 == key {
            continue;
        }
        // Tied block occupies 0-based positions s..e, with p relevant items.
        let e = i + 1;
        let t = e - s;
        let (tf, pf) = (t as f64, p as f64);
        let nf = tf - pf;
        roc += nf * (tp + tp + pf) * 0.5;

        if s < kmax {
            for (j, &k) in ks.iter().enumerate() {
                if s >= k {
                    continue;
                }
                let a = e.min(k) - s; // block positions inside the cutoff
                let af = a as f64;
                sc.dcg[j] += gsum * (tab.disc[s + a] - tab.disc[s]) / tf;
                if p == 0 {
                    continue;
                }
                sc.tpk[j] += pf * af / tf;

                // AP: position s+q is relevant w.p. p/t; given that, the expected
                // number of relevant items at or above it is
                // tp + 1 + (q-1)(p-1)/(t-1).
                let h = tab.harm[s + a] - tab.harm[s];
                let mut term = (tp + 1.0) * h;
                if t > 1 {
                    term += (pf - 1.0) / (tf - 1.0) * (af - (s as f64 + 1.0) * h);
                }
                sc.ap[j] += pf / tf * term;

                // MRR: only the first block holding a relevant item matters.
                // P(first relevant at block offset q) = C(t-q, p-1) / C(t, p).
                if tp == 0.0 {
                    let lim = a.min(t - p + 1);
                    let mut pq = pf / tf;
                    let mut acc = 0.0;
                    for q in 1..=lim {
                        acc += pq / (s + q) as f64;
                        if q < lim {
                            pq *= (t - q - p + 1) as f64 / (t - q) as f64;
                        }
                    }
                    sc.mrr[j] = acc;
                }

                // Hit rate: P(none of the first `a` block slots is relevant).
                if a > t - p {
                    sc.miss[j] = 0.0;
                } else if sc.miss[j] > 0.0 {
                    let mut pr = 1.0;
                    for q in 0..a {
                        pr *= (t - p - q) as f64 / (t - q) as f64;
                    }
                    sc.miss[j] *= pr;
                }
            }
        }

        tp += pf;
        fp += nf;
        s = e;
        p = 0;
        gsum = 0.0;
    }

    if tp > 0.0 && fp > 0.0 {
        out[0] = roc / (tp * fp);
    }
    let n_rel = judged.as_ref().map_or(tp, |j| j.n_relevant);
    if n_rel == 0.0 {
        return; // no relevant items: ranking metrics undefined
    }

    // Ideal DCG at each cutoff, unless the judgments supply it.
    if judged.is_none() && !opts.binary {
        sc.ideal.clear();
        sc.ideal.extend(m.iter().map(|&i| gains[i as usize]));
        if kmax < len {
            sc.ideal.select_nth_unstable_by(kmax, |a, b| b.total_cmp(a));
            sc.ideal.truncate(kmax);
        }
        sc.ideal.sort_unstable_by(|a, b| b.total_cmp(a));
        let mut acc = 0.0;
        for (r, g) in sc.ideal.iter_mut().enumerate() {
            acc += *g * (tab.disc[r + 1] - tab.disc[r]);
            *g = acc;
        }
    }

    for (j, &k) in ks.iter().enumerate() {
        let keff = k.min(len);
        let o = &mut out[1 + j * PER_K..1 + (j + 1) * PER_K];
        let idcg = match &judged {
            Some(jd) => jd.idcg[j],
            None if opts.binary => tab.disc[(tp as usize).min(keff)],
            None => sc.ideal[keff - 1],
        };
        o[0] = if idcg > 0.0 { sc.dcg[j] / idcg } else { f64::NAN };
        o[1] = sc.mrr[j];
        let map_den = if opts.map_by_total { n_rel } else { n_rel.min(k as f64) };
        o[2] = sc.ap[j] / map_den;
        o[3] = sc.tpk[j] / n_rel;
        o[4] = sc.tpk[j] / if k == usize::MAX { len as f64 } else { k as f64 };
        o[5] = 1.0 - sc.miss[j];
    }
}

/// Grouped ranking metrics.
///
/// Returns `(sums, weights, extra, per_group)`:
/// * `sums[c]` / `weights[c]`: group-weighted sum of column `c` and total
///   group weight, over groups where the column is defined;
/// * `extra = [auc_sum, auc_weight, n_nonempty, n_ranked, n_auc, w_nonempty]`
///   where the AUC aggregate weights each group by `group_weight * size`;
/// * `per_group`: flat `(n_groups, 1 + 6 * len(ks))` matrix, empty unless requested.
///
/// `judged_n_relevant` (n_groups) and `judged_idcg` (n_groups x len(ks)), when
/// given, describe the complete judgments for each group.
#[pyfunction]
#[pyo3(signature = (y_true, gains, y_score, codes, n_groups, ks, per_group, group_weight=None,
                    binary=false, map_by_total=false, judged_n_relevant=None, judged_idcg=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ranking<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    gains: PyReadonlyArray1<'py, f64>,
    y_score: PyReadonlyArray1<'py, f64>,
    codes: PyReadonlyArray1<'py, i64>,
    n_groups: usize,
    ks: Vec<usize>,
    per_group: bool,
    group_weight: Option<PyReadonlyArray1<'py, f64>>,
    binary: bool,
    map_by_total: bool,
    judged_n_relevant: Option<PyReadonlyArray1<'py, f64>>,
    judged_idcg: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let yt = y_true.as_slice()?;
    let gains = gains.as_slice()?;
    let ys = y_score.as_slice()?;
    let codes = codes.as_slice()?;
    let gw = weights_slice(&group_weight)?;
    let jn = weights_slice(&judged_n_relevant)?;
    let ji = weights_slice(&judged_idcg)?;
    let n = ys.len();
    check_len(n)?;
    check_same("y_true / y_score", yt.len(), n)?;
    check_same("gains / y_score", gains.len(), n)?;
    check_same("groups / y_score", codes.len(), n)?;
    check_scores(ys)?;
    check_weights(gw, n_groups)?;
    if gains.iter().any(|g| !g.is_finite() || *g < 0.0) {
        return err("relevance gains must be finite and non-negative");
    }
    if codes.iter().any(|&c| c < 0 || c as u64 >= n_groups as u64) {
        return err("group codes must lie in [0, n_groups)");
    }
    if ks.contains(&0) {
        return err("cutoffs must be >= 1");
    }
    if jn.is_some() != ji.is_some() {
        return err("judged_n_relevant and judged_idcg go together");
    }
    if let (Some(jn), Some(ji)) = (jn, ji) {
        check_same("judged_n_relevant / groups", jn.len(), n_groups)?;
        check_same("judged_idcg / groups x cutoffs", ji.len(), n_groups * ks.len())?;
        if jn.iter().chain(ji.iter()).any(|v| !v.is_finite() || *v < 0.0) {
            return err("judgments must be finite and non-negative");
        }
    }
    let nk = ks.len();
    let width = 1 + PER_K * nk;

    const CHUNK: usize = 512;
    const EXTRA: usize = 6;
    let outcome = detached(py, || {
        let mut start = vec![0usize; n_groups + 1];
        for &c in codes {
            start[c as usize + 1] += 1;
        }
        let mut max_len = 0;
        for g in 0..n_groups {
            max_len = max_len.max(start[g + 1]);
            start[g + 1] += start[g];
        }
        // Rows of one group are usually contiguous already; then the
        // members list is just 0..n and needs no scatter.
        let members: Vec<u32> = if codes.windows(2).all(|w| w[0] <= w[1]) {
            (0..n as u32).collect()
        } else {
            let mut cursor = start.clone();
            let mut members = vec![0u32; n];
            for (i, &c) in codes.iter().enumerate() {
                let slot = &mut cursor[c as usize];
                members[*slot] = i as u32;
                *slot += 1;
            }
            members
        };
        // In-list relevant items may not exceed the judged total.
        if let Some(jn) = jn {
            for g in 0..n_groups {
                let in_list = members[start[g]..start[g + 1]]
                    .iter()
                    .filter(|&&i| yt[i as usize] != 0)
                    .count();
                if in_list as f64 > jn[g] {
                    return Err(g);
                }
            }
        }
        let tab = RankTables::new(max_len);
        let opts = RankOpts { ks: &ks, binary, map_by_total };

        // Fixed chunks, reduced in order: bit-identical for any thread count.
        struct Part {
            sums: Vec<f64>,
            wts: Vec<f64>,
            extra: [f64; EXTRA],
            rows: Vec<f64>,
        }
        let parts: Vec<Part> = (0..n_groups.div_ceil(CHUNK))
            .into_par_iter()
            .map_init(Scratch::default, |sc, c| {
                let lo = c * CHUNK;
                let hi = (lo + CHUNK).min(n_groups);
                let mut part = Part {
                    sums: vec![0.0; width],
                    wts: vec![0.0; width],
                    extra: [0.0; EXTRA],
                    rows: Vec::with_capacity(if per_group { (hi - lo) * width } else { 0 }),
                };
                let mut buf = vec![0.0f64; width];
                for g in lo..hi {
                    let m = &members[start[g]..start[g + 1]];
                    let qw = gw.map_or(1.0, |w| w[g]);
                    let judged = match (jn, ji) {
                        (Some(jn), Some(ji)) => Some(Judged {
                            n_relevant: jn[g],
                            idcg: &ji[g * nk..(g + 1) * nk],
                        }),
                        _ => None,
                    };
                    rank_group(m, yt, gains, ys, &opts, judged, &tab, sc, &mut buf);
                    if !m.is_empty() {
                        part.extra[2] += 1.0;
                        part.extra[5] += qw;
                    }
                    if !buf[0].is_nan() {
                        let aw = qw * m.len() as f64;
                        part.extra[0] += buf[0] * aw;
                        part.extra[1] += aw;
                        part.extra[4] += 1.0;
                    }
                    if width > 1 && !buf[2].is_nan() {
                        part.extra[3] += 1.0;
                    }
                    for (c, &v) in buf.iter().enumerate() {
                        if !v.is_nan() {
                            part.sums[c] += qw * v;
                            part.wts[c] += qw;
                        }
                    }
                    if per_group {
                        part.rows.extend_from_slice(&buf);
                    }
                }
                part
            })
            .collect();

        let mut sums = vec![0.0; width];
        let mut wts = vec![0.0; width];
        let mut extra = vec![0.0f64; EXTRA];
        let mut rows = Vec::with_capacity(if per_group { n_groups * width } else { 0 });
        for part in parts {
            for c in 0..width {
                sums[c] += part.sums[c];
                wts[c] += part.wts[c];
            }
            for (e, p) in extra.iter_mut().zip(part.extra) {
                *e += p;
            }
            rows.extend(part.rows);
        }
        Ok((sums, wts, extra, rows))
    });

    let (sums, wts, extra, rows) = match outcome {
        Ok(v) => v,
        Err(g) => {
            return err(format!(
                "group {g} has more relevant items in the list than in the judgments"
            ))
        }
    };
    Ok((
        sums.into_pyarray(py),
        wts.into_pyarray(py),
        extra.into_pyarray(py),
        rows.into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------------
// DeLong structural components
// ---------------------------------------------------------------------------

/// DeLong structural components for K score rows (K, N) against one label vector.
///
/// For positive `i`, `v10[i]` is the fraction of negatives scored below it
/// (ties count half); for negative `j`, `v01[j]` is the fraction of positives
/// scored above it. Both average to the ROC-AUC. Positives and negatives keep
/// their original relative order, so rows line up across models.
///
/// Returns flat arrays of shape (K, n_pos) and (K, n_neg).
#[pyfunction]
fn delong_components<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_scores: PyReadonlyArray2<'py, f64>,
) -> PyResult<ArrPair<'py>> {
    let yt = y_true.as_slice()?;
    let ys: ArrayView2<f64> = y_scores.as_array();
    let n = yt.len();
    check_len(n)?;
    check_same("y_true / y_score", n, ys.ncols())?;
    check_scores(ys.iter())?;
    let n_pos = yt.iter().filter(|&&y| y != 0).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return err("DeLong needs both classes");
    }

    let (v10, v01) = detached(py, || {
        // Position of each sample within its own class.
        let mut slot = vec![0u32; n];
        let (mut p, mut q) = (0u32, 0u32);
        for (i, &y) in yt.iter().enumerate() {
            if y != 0 {
                slot[i] = p;
                p += 1;
            } else {
                slot[i] = q;
                q += 1;
            }
        }
        let rows: Vec<(Vec<f64>, Vec<f64>)> = (0..ys.nrows())
            .into_par_iter()
            .map(|r| {
                let s = sort_desc(ys.row(r).iter().copied());
                let mut v10 = vec![0.0f64; n_pos];
                let mut v01 = vec![0.0f64; n_neg];
                let (mut pos_above, mut neg_seen, mut start) = (0usize, 0usize, 0usize);
                for i in 0..n {
                    if !s.tie_end[i] {
                        continue;
                    }
                    let block = &s.order[start..=i];
                    let pb = block.iter().filter(|&&j| yt[j as usize] != 0).count();
                    let nb = block.len() - pb;
                    let neg_below = n_neg - neg_seen - nb;
                    let a = (neg_below as f64 + 0.5 * nb as f64) / n_neg as f64;
                    let b = (pos_above as f64 + 0.5 * pb as f64) / n_pos as f64;
                    for &j in block {
                        let j = j as usize;
                        if yt[j] != 0 {
                            v10[slot[j] as usize] = a;
                        } else {
                            v01[slot[j] as usize] = b;
                        }
                    }
                    pos_above += pb;
                    neg_seen += nb;
                    start = i + 1;
                }
                (v10, v01)
            })
            .collect();
        let mut v10 = Vec::with_capacity(rows.len() * n_pos);
        let mut v01 = Vec::with_capacity(rows.len() * n_neg);
        for (a, b) in rows {
            v10.extend(a);
            v01.extend(b);
        }
        (v10, v01)
    });
    Ok(to_py(py, (v10, v01)))
}

// ---------------------------------------------------------------------------
// Exact hash grouping
// ---------------------------------------------------------------------------

/// Dense codes for fixed-width ids (one row of bytes per sample), by hashing.
///
/// Exact: the hash only picks a bucket; ids are compared byte for byte, so
/// two different ids are never merged, however their hashes collide.
/// Returns (codes, first) where `codes[i]` numbers ids in order of first
/// appearance and `first[c]` is the index of the first sample with code `c`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn hash_codes<'py>(
    py: Python<'py>,
    ids: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let a: ArrayView2<u8> = ids.as_array();
    let n = a.nrows();
    let width = a.ncols();
    let flat = match a.as_slice() {
        Some(s) => s,
        None => return err("ids must be C-contiguous"),
    };
    let (codes, first) = detached(py, || {
        let mut map: rustc_hash::FxHashMap<&[u8], i64> = rustc_hash::FxHashMap::default();
        map.reserve(n.min(1 << 20));
        let mut codes = Vec::with_capacity(n);
        let mut first = Vec::new();
        for i in 0..n {
            let key = &flat[i * width..(i + 1) * width];
            let next = first.len() as i64;
            let code = *map.entry(key).or_insert_with(|| {
                first.push(i as i64);
                next
            });
            codes.push(code);
        }
        (codes, first)
    });
    Ok((codes.into_pyarray(py), first.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Score tables (exact streaming)
// ---------------------------------------------------------------------------

type Table<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Collapse a batch to its distinct scores, in descending order, with the
/// total positive and negative weight at each score.
///
/// Every threshold metric depends on the data only through this table
/// (samples with equal scores are interchangeable), so metrics computed from
/// it are exactly the metrics of the rows it summarises.
#[pyfunction]
#[pyo3(signature = (y_true, y_score, sample_weight=None))]
fn score_table<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<'py, u8>,
    y_score: PyReadonlyArray1<'py, f64>,
    sample_weight: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Table<'py>> {
    let yt = y_true.as_slice()?;
    let ys = y_score.as_slice()?;
    let w = weights_slice(&sample_weight)?;
    check_same("y_true / y_score", yt.len(), ys.len())?;
    check_len(ys.len())?;
    check_scores(ys)?;
    check_weights(w, ys.len())?;
    let (s, p, n) = detached(py, || {
        let sorted = sort_desc(ys.iter().copied());
        let (mut s, mut p, mut n) = (Vec::new(), Vec::new(), Vec::new());
        let (mut tp, mut fp) = (0.0f64, 0.0f64);
        for (&idx, &end) in sorted.order.iter().zip(sorted.tie_end.iter()) {
            let i = idx as usize;
            let wi = w.map_or(1.0, |w| w[i]);
            if yt[i] != 0 {
                tp += wi;
            } else {
                fp += wi;
            }
            if end {
                s.push(ys[i] + 0.0); // one representative; -0.0 becomes +0.0
                p.push(tp);
                n.push(fp);
                tp = 0.0;
                fp = 0.0;
            }
        }
        (s, p, n)
    });
    Ok((s.into_pyarray(py), p.into_pyarray(py), n.into_pyarray(py)))
}

/// Merge two score tables (each descending, distinct scores) in linear time.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn merge_score_tables<'py>(
    py: Python<'py>,
    a_score: PyReadonlyArray1<'py, f64>,
    a_pos: PyReadonlyArray1<'py, f64>,
    a_neg: PyReadonlyArray1<'py, f64>,
    b_score: PyReadonlyArray1<'py, f64>,
    b_pos: PyReadonlyArray1<'py, f64>,
    b_neg: PyReadonlyArray1<'py, f64>,
) -> PyResult<Table<'py>> {
    let (sa, pa, na) = (a_score.as_slice()?, a_pos.as_slice()?, a_neg.as_slice()?);
    let (sb, pb, nb) = (b_score.as_slice()?, b_pos.as_slice()?, b_neg.as_slice()?);
    check_same("table a", sa.len(), pa.len())?;
    check_same("table a", sa.len(), na.len())?;
    check_same("table b", sb.len(), pb.len())?;
    check_same("table b", sb.len(), nb.len())?;
    check_scores(sa)?;
    check_scores(sb)?;
    let strictly_desc = |s: &[f64]| s.windows(2).all(|w| desc_key(w[0]) < desc_key(w[1]));
    if !strictly_desc(sa) || !strictly_desc(sb) {
        return err("score tables must hold distinct scores in descending order");
    }
    let (s, p, n) = detached(py, || {
        let cap = sa.len() + sb.len();
        let (mut s, mut p, mut n) =
            (Vec::with_capacity(cap), Vec::with_capacity(cap), Vec::with_capacity(cap));
        let (mut i, mut j) = (0, 0);
        while i < sa.len() || j < sb.len() {
            let ka = sa.get(i).map_or(u64::MAX, |&x| desc_key(x));
            let kb = sb.get(j).map_or(u64::MAX, |&x| desc_key(x));
            if ka < kb || j == sb.len() {
                s.push(sa[i] + 0.0);
                p.push(pa[i]);
                n.push(na[i]);
                i += 1;
            } else if kb < ka || i == sa.len() {
                s.push(sb[j] + 0.0);
                p.push(pb[j]);
                n.push(nb[j]);
                j += 1;
            } else {
                s.push(sa[i] + 0.0);
                p.push(pa[i] + pb[j]);
                n.push(na[i] + nb[j]);
                i += 1;
                j += 1;
            }
        }
        (s, p, n)
    });
    Ok((s.into_pyarray(py), p.into_pyarray(py), n.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Paired randomization test
// ---------------------------------------------------------------------------

/// Sign-flip randomization test on paired differences `d`.
///
/// Under the null hypothesis each difference is equally likely to have either
/// sign. Counts the sign patterns whose sum is at least as extreme as the
/// observed sum: two-sided (`|S| >= |S_obs|`), greater (`S >= S_obs`) and
/// less (`S <= S_obs`). When `2**n <= n_resamples` every pattern is
/// enumerated and the test is exact; otherwise `n_resamples` random
/// patterns are drawn. Counts are integers, so the result does not depend on
/// the thread count.
///
/// Returns (patterns, two_sided, greater, less, exact).
#[pyfunction]
fn sign_flip(
    py: Python<'_>,
    d: PyReadonlyArray1<'_, f64>,
    n_resamples: u64,
    seed: u64,
) -> PyResult<(u64, u64, u64, u64, bool)> {
    let d = d.as_slice()?;
    if d.iter().any(|x| !x.is_finite()) {
        return err("paired differences must be finite");
    }
    let n = d.len();
    let obs: f64 = d.iter().sum();
    // Patterns that tie the observed sum up to rounding count as "as extreme".
    let tol = 1e-12 * d.iter().map(|x| x.abs()).sum::<f64>();
    // Enumeration is O(n 2^n); beyond 2^30 patterns sampling is the only option.
    let exact = n <= 30 && (1u64 << n) <= n_resamples;
    let tally = move |s: f64| -> [u64; 3] {
        [
            (s.abs() >= obs.abs() - tol) as u64,
            (s >= obs - tol) as u64,
            (s <= obs + tol) as u64,
        ]
    };
    let add = |a: [u64; 3], b: [u64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let counts = detached(py, || {
        if exact {
            (0..1usize << n)
                .into_par_iter()
                .with_min_len(1 << 10)
                .map(|mask| {
                    let s: f64 = d
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| if (mask >> i) & 1 == 1 { -x } else { x })
                        .sum();
                    tally(s)
                })
                .reduce(|| [0; 3], add)
        } else {
            (0..n_resamples as usize)
                .into_par_iter()
                .with_min_len(16)
                .map(|r| {
                    let mut rng = Rng::new(seed, r as u64);
                    let (mut s, mut bits, mut left) = (0.0f64, 0u64, 0u32);
                    for &x in d {
                        if left == 0 {
                            bits = rng.next_u64();
                            left = 64;
                        }
                        s += if bits & 1 == 1 { -x } else { x };
                        bits >>= 1;
                        left -= 1;
                    }
                    tally(s)
                })
                .reduce(|| [0; 3], add)
        }
    });
    let patterns = if exact { 1u64 << n } else { n_resamples };
    Ok((patterns, counts[0], counts[1], counts[2], exact))
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_table, m)?)?;
    m.add_function(wrap_pyfunction!(merge_score_tables, m)?)?;
    m.add_function(wrap_pyfunction!(sign_flip, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc_ap, m)?)?;
    m.add_function(wrap_pyfunction!(many_scores, m)?)?;
    m.add_function(wrap_pyfunction!(many_targets, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise, m)?)?;
    m.add_function(wrap_pyfunction!(grouped, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(binary_summary, m)?)?;
    m.add_function(wrap_pyfunction!(curve, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_sums, m)?)?;
    m.add_function(wrap_pyfunction!(ranking, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_group_means, m)?)?;
    m.add_function(wrap_pyfunction!(delong_components, m)?)?;
    m.add_function(wrap_pyfunction!(hash_codes, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
