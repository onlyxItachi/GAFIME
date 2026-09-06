//! Nonshipping native candidate-bank / existing Core-CUDA scorer comparison.
//! New transforms execute in Rust; CUDA receives ordinary unary columns. This
//! does not advertise CUDA transformation, paired-view, or graph-evidence APIs.

#[allow(dead_code)] // Shared fixture module has APIs used by its own unit tests.
mod issue73_candidates;
#[allow(dead_code)] // Direct-product controls are optional for non-product cells.
mod issue73_gpu_reuse;

use issue73_candidates::{
    materialize_row_major, planted_fixture, planted_invariance_fixture,
    select_train_view_consistency, synthetic_inputs, CandidateBank, CandidateCatalog, CandidateOp,
    NativeMatrixRef,
};
use issue73_gpu_reuse::Issue73UnaryScorer;
use std::{env, hint::black_box, time::Instant};

fn failure(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

// Independent row-ordered f64 statistical oracle, with f32 pointwise values
// supplied by the bank. It does not call a GAFIME correlation primitive.
fn oracle(bank: &CandidateBank, target: &[f32]) -> Vec<f64> {
    (0..bank.candidate_count)
        .map(|column| {
            let n = bank.rows as f64;
            let mx = (0..bank.rows)
                .map(|row| f64::from(bank.values_row_major[row * bank.candidate_count + column]))
                .sum::<f64>()
                / n;
            let my = target.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
            for (row, &y) in target.iter().enumerate() {
                let x = f64::from(bank.values_row_major[row * bank.candidate_count + column]) - mx;
                let y = f64::from(y) - my;
                xx += x * x;
                xy += x * y;
                yy += y * y;
            }
            if xx == 0.0 || yy == 0.0 {
                0.0
            } else {
                xy / (xx * yy).sqrt()
            }
        })
        .collect()
}

fn parity(actual: &[f64], expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err("candidate count mismatch".into());
    }
    let mut largest: f64 = 0.0;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            return Err("non-finite fixture score".into());
        }
        largest = largest.max((a - e).abs());
    }
    // Existing Core mixed regrouping bound; tested experimentally for this
    // finite mixed CUDA slice too, not a newly relaxed production tolerance.
    if largest > 1e-12 {
        return Err(format!("mixed scalar-oracle error {largest} > 1e-12"));
    }
    Ok(largest)
}

fn verify_backend(scorer: &Issue73UnaryScorer, requested: &str) -> Result<(), String> {
    let identity = scorer.backend_identity();
    if identity.backend != requested {
        return Err("backend identity mismatch".into());
    }
    if requested == "cuda" {
        let loaded = identity
            .library_path
            .as_ref()
            .ok_or("CUDA path not reported")?;
        let intended = env::var("GAFIME_CUDA_V1_LIB").map_err(failure)?;
        if std::fs::canonicalize(loaded).map_err(failure)?
            != std::fs::canonicalize(intended).map_err(failure)?
        {
            return Err("CUDA loaded a different payload".into());
        }
        if scorer
            .last_execution()
            .ok_or("CUDA execution missing")?
            .launched_chunks
            == 0
        {
            return Err("CUDA reported no launched chunks".into());
        }
    }
    Ok(())
}

fn quality(backend: &str) -> Result<(), String> {
    let mut records = Vec::new();
    for control in ["identity", "invariance"] {
        for seed in 73..78 {
            let fixture = if control == "identity" {
                planted_fixture(seed)
            } else {
                planted_invariance_fixture(seed)
            }
            .map_err(failure)?;
            let bank = |data: &[f32], rows| {
                let input =
                    NativeMatrixRef::new(rows, fixture.feature_count, data).map_err(failure)?;
                materialize_row_major(&input, &fixture.catalog).map_err(failure)
            };
            let train = bank(&fixture.train_primary, fixture.train_rows)?;
            let aligned = bank(&fixture.train_aligned, fixture.train_rows)?;
            let shuffled = bank(&fixture.train_shuffled, fixture.train_rows)?;
            let selection = select_train_view_consistency(&train, &aligned).map_err(failure)?;
            let negative = select_train_view_consistency(&train, &shuffled).map_err(failure)?;
            // Holdout data and labels are consulted only after the index is frozen.
            let holdout = bank(&fixture.holdout_primary, fixture.holdout_rows)?;
            let expected = oracle(&holdout, &fixture.holdout_target);
            let mut scorer = Issue73UnaryScorer::prepare(
                backend,
                holdout.rows,
                holdout.candidate_count,
                &holdout.values_row_major,
                &fixture.holdout_target,
            )?;
            let actual = scorer.execute_resident()?;
            verify_backend(&scorer, backend)?;
            let max_error = parity(&actual, &expected)?;
            let selected = selection.selected_index;
            if control == "invariance" && selected != 1 {
                return Err("invariance control did not select absolute difference".into());
            }
            if actual[selected].abs() < 0.9 {
                return Err("planted holdout signal was not recovered".into());
            }
            records.push(format!(
            "{{\"control\":\"{control}\",\"seed\":{seed},\"selected_index\":{selected},\"train_view_scores\":{:?},\
             \"shuffled_view_scores\":{:?},\"holdout_scores\":{:?},\"max_oracle_error\":{max_error}}}",
            selection.per_candidate_scores, negative.per_candidate_scores, actual));
        }
    }
    println!(
        "{{\"backend\":\"{backend}\",\"precision\":\"mixed\",\
              \"selection_backend\":\"core\",\"selection_has_target\":false,\
              \"quality_scope\":\"planted invariance control, not learner efficacy\",\
              \"records\":[{}]}}",
        records.join(",")
    );
    Ok(())
}

fn bench(backend: &str, args: &[String]) -> Result<(), String> {
    if args.len() != 6 {
        return Err("bench requires rows count kind warmups samples min_ms".into());
    }
    let parse = |index: usize| -> Result<usize, String> {
        let value = args[index].parse::<usize>().map_err(failure)?;
        if value == 0 {
            return Err("benchmark controls must be positive".into());
        }
        Ok(value)
    };
    let (rows, count, warmups, repetitions, sample_ms) =
        (parse(0)?, parse(1)?, parse(3)?, parse(4)?, parse(5)?);
    if rows < 2
        || rows > 32768
        || count > 256
        || warmups > 100
        || repetitions > 100
        || sample_ms > 1000
    {
        return Err("outside bounded experiment size/repetition envelope".into());
    }
    let kind = args[2].as_str();
    if !matches!(kind, "absdiff" | "product" | "product_direct") {
        return Err("unknown candidate kind".into());
    }
    let mut columns = 2usize;
    while columns * (columns - 1) / 2 < count {
        columns += 1;
    }
    let mut operations = Vec::new();
    for left in 0..columns {
        for right in left + 1..columns {
            if operations.len() == count {
                break;
            }
            operations.push(if kind == "absdiff" {
                CandidateOp::AbsoluteDifference { left, right }
            } else {
                CandidateOp::CenteredProduct2 { left, right }
            });
        }
    }
    let catalog = CandidateCatalog::new(columns, operations).map_err(failure)?;
    let input_values = synthetic_inputs(rows, columns, 73).map_err(failure)?;
    let input = NativeMatrixRef::new(rows, columns, &input_values).map_err(failure)?;
    // A measured feature column is the common reference/anchor, not fabricated
    // labels. The current ABI calls this row-aligned operand `target`.
    let anchor: Vec<f32> = input_values
        .chunks_exact(columns)
        .map(|row| row[0])
        .collect();
    // Inputs/catalog/anchor already exist. This is preparation plus the first
    // execution, not process-cold end-to-end latency. The direct control also
    // follows oracle-only bank materialization, which can warm allocator/cache.
    let mut preparation_start = Instant::now();
    let start = Instant::now();
    let bank = materialize_row_major(&input, &catalog).map_err(failure)?;
    let materialize_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let direct = kind == "product_direct";
    if direct && columns * (columns - 1) / 2 != count {
        return Err("direct product control requires a complete all-pairs catalog".into());
    }
    if direct {
        preparation_start = Instant::now();
    }
    let mut scorer = if direct {
        Issue73UnaryScorer::prepare_existing_centered_products(
            backend,
            rows,
            columns,
            &input_values,
            &anchor,
            &(0..columns as u32).collect::<Vec<_>>(),
        )?
    } else {
        Issue73UnaryScorer::prepare(backend, rows, count, &bank.values_row_major, &anchor)?
    };
    let setup_ns = start.elapsed().as_nanos();
    let actual = scorer.execute_resident()?;
    verify_backend(&scorer, backend)?;
    let preparation_and_first_execute_ns = preparation_start.elapsed().as_nanos();
    let expected = oracle(&bank, &anchor);
    let max_error = parity(&actual, &expected)?;
    for _ in 0..warmups {
        scorer.execute_resident_in_place()?;
    }
    let mut loops = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..loops {
            scorer.execute_resident_in_place()?;
        }
        if start.elapsed().as_millis() >= sample_ms as u128 || loops >= 65536 {
            break;
        }
        loops *= 2;
    }
    let mut samples = Vec::with_capacity(repetitions);
    let mut region_ns = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        for _ in 0..loops {
            scorer.execute_resident_in_place()?;
        }
        let duration = start.elapsed().as_nanos();
        samples.push(duration as f64 / loops as f64);
        region_ns.push(duration);
        black_box(scorer.output_values()?);
    }
    // Exact repeat identity is checked outside the timed loop. Cross-backend
    // floating parity and within-backend repeat determinism are distinct gates.
    if scorer.execute_resident()? != actual {
        return Err("repeat result identity changed".into());
    }
    println!(
        "{{\"backend\":\"{backend}\",\"precision\":\"mixed\",\
              \"candidate_kind\":\"{kind}\",\"rows\":{rows},\"candidates\":{count},\
              \"source_columns\":{columns},\"rayon_workers\":{},\
              \"materialize_ns_single_observation\":{materialize_ns},\
              \"setup_ns_single_observation\":{setup_ns},\
              \"preparation_and_first_execute_ns_single_observation\":{preparation_and_first_execute_ns},\
              \"prior_oracle_bank_materialization_excluded\":{direct},\
              \"execution_materialized_bank_bytes\":{},\"oracle_bank_bytes\":{},\
              \"materialization_peak_value_bytes\":{},\
              \"warmups\":{warmups},\"loops_per_sample\":{loops},\
              \"resident_ns_per_call\":{samples:?},\"measured_region_ns\":{region_ns:?},\
              \"max_oracle_error\":{max_error},\"scores\":{actual:?},\
              \"generation_backend\":\"{}\",\"scoring_backend\":\"{backend}\"}}",
        rayon::current_num_threads(),
        if direct { 0 } else { bank.row_major_bytes },
        bank.row_major_bytes,
        bank.conservative_peak_value_bytes,
        if direct { backend } else { "core" }
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.as_slice() {
        [mode, backend] if mode == "quality" => quality(backend),
        [mode, backend, rest @ ..] if mode == "bench" => bench(backend, rest),
        _ => Err(
            "use quality core|cuda OR bench core|cuda rows count kind warmups samples min_ms"
                .into(),
        ),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_checks_values_and_count() {
        assert!(parity(&[0.2], &[0.2]).is_ok());
        assert!(parity(&[f64::NAN], &[0.2]).is_err());
        assert!(parity(&[], &[0.2]).is_err());
        assert!(parity(&[0.3], &[0.2]).is_err());
    }

    #[test]
    fn benchmark_controls_fail_closed() {
        let args = ["0", "16", "absdiff", "10", "30", "100"].map(String::from);
        assert!(bench("core", &args).is_err());
    }
}
