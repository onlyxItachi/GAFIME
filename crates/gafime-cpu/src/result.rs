use gafime_types::{
    GafimeResultTable, GafimeResultTableF64, GAFIME_ABI_VERSION, GAFIME_PRECISION_ABI_VERSION,
};

use gafime_orchestrator::{OrchestratorError, OrchestratorResult};

use crate::{
    precision::{
        CpuDtype, CpuPrecisionSlice, CpuPrecisionValues, PrecisionProfile, PrecisionProfileExt,
    },
    rank::top_k_precision_indices,
};

/// Safe, profile-typed Core result ownership.
///
/// The legacy [`OwnedResultTable`] remains available for ABI 1.0 f32 callers.
/// This container is the matching Rust surface for the versioned typed result
/// ABI: `mixed` and `fp64` retain f64 scores through ranking and public-table
/// ownership instead of writing them into the old `*mut f32` field.
#[derive(Debug, PartialEq)]
pub struct CpuPrecisionResultTable {
    profile: PrecisionProfile,
    max_arity: u32,
    metric_count: u32,
    capacity: u64,
    row_count: u64,
    combo_indices: Vec<u32>,
    metric_values: CpuPrecisionValues,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl CpuPrecisionResultTable {
    pub fn new(
        profile: PrecisionProfile,
        capacity: u64,
        max_arity: u32,
        metric_count: u32,
    ) -> OrchestratorResult<Self> {
        let capacity_usize = usize::try_from(capacity).map_err(|_| {
            OrchestratorError::InvalidPlan("CPU precision result capacity exceeds address space")
        })?;
        let combo_len = capacity_usize.checked_mul(max_arity as usize).ok_or(
            OrchestratorError::InvalidPlan("CPU precision result combo storage overflows"),
        )?;
        let metric_len = capacity_usize.checked_mul(metric_count as usize).ok_or(
            OrchestratorError::InvalidPlan("CPU precision result metric storage overflows"),
        )?;
        let metric_values = match profile.cpu_contract().ranking_and_public_result {
            CpuDtype::F32 => CpuPrecisionValues::F32(vec![0.0; metric_len]),
            CpuDtype::F64 => CpuPrecisionValues::F64(vec![0.0; metric_len]),
        };
        Ok(Self {
            profile,
            max_arity,
            metric_count,
            capacity,
            row_count: 0,
            combo_indices: vec![u32::MAX; combo_len],
            metric_values,
            ranks: vec![0; capacity_usize],
            families: vec![0; capacity_usize],
            candidate_ids: vec![0; capacity_usize],
            row_flags: vec![0; capacity_usize],
        })
    }

    pub fn profile(&self) -> PrecisionProfile {
        self.profile
    }

    pub fn profile_identity(&self) -> u32 {
        self.profile.profile_identity()
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn max_arity(&self) -> u32 {
        self.max_arity
    }

    pub fn metric_count(&self) -> u32 {
        self.metric_count
    }

    pub fn metric_values(&self) -> &CpuPrecisionValues {
        &self.metric_values
    }

    pub fn combo_indices(&self) -> &[u32] {
        &self.combo_indices
    }

    pub fn ranks(&self) -> &[u32] {
        &self.ranks[..self.row_count as usize]
    }

    pub fn candidate_ids(&self) -> &[u64] {
        &self.candidate_ids[..self.row_count as usize]
    }

    pub fn families(&self) -> &[u32] {
        &self.families[..self.row_count as usize]
    }

    pub fn row_flags(&self) -> &[u32] {
        &self.row_flags[..self.row_count as usize]
    }

    /// Append a complete row.  Structural candidate metadata stays integer;
    /// `scores` must use the exact public result dtype of this table's profile.
    #[allow(clippy::too_many_arguments)]
    pub fn push_row(
        &mut self,
        combo: &[u32],
        scores: CpuPrecisionSlice<'_>,
        family: u32,
        candidate_id: u64,
        flags: u32,
    ) -> OrchestratorResult<()> {
        if self.row_count == self.capacity {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision result table capacity is exhausted",
            ));
        }
        if combo.len() > self.max_arity as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision result combo exceeds max arity",
            ));
        }
        if scores.len() != self.metric_count as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision result metric width does not match table",
            ));
        }
        let expected_dtype = self.profile.cpu_contract().ranking_and_public_result;
        if scores.dtype() != expected_dtype {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision result dtype does not match its profile",
            ));
        }
        let row = self.row_count as usize;
        let combo_offset = row * self.max_arity as usize;
        self.combo_indices[combo_offset..combo_offset + self.max_arity as usize].fill(u32::MAX);
        self.combo_indices[combo_offset..combo_offset + combo.len()].copy_from_slice(combo);
        let metric_offset = row * self.metric_count as usize;
        match (&mut self.metric_values, scores) {
            (CpuPrecisionValues::F32(destination), CpuPrecisionSlice::F32(source)) => {
                destination[metric_offset..metric_offset + source.len()].copy_from_slice(source);
            }
            (CpuPrecisionValues::F64(destination), CpuPrecisionSlice::F64(source)) => {
                destination[metric_offset..metric_offset + source.len()].copy_from_slice(source);
            }
            _ => {
                return Err(OrchestratorError::InvalidPlan(
                    "CPU precision result values changed dtype during append",
                ));
            }
        }
        self.ranks[row] = u32::try_from(row)
            .map_err(|_| OrchestratorError::InvalidPlan("CPU precision result rank exceeds u32"))?;
        self.families[row] = family;
        self.candidate_ids[row] = candidate_id;
        self.row_flags[row] = flags;
        self.row_count += 1;
        Ok(())
    }

    /// Stable in-place top-k selection based on the table's visible score lane.
    pub fn compact_top_k(
        &mut self,
        metric_index: usize,
        descending: bool,
        top_k: usize,
    ) -> OrchestratorResult<u64> {
        if metric_index >= self.metric_count as usize {
            return Err(OrchestratorError::InvalidPlan(
                "CPU precision rank metric index exceeds result metric count",
            ));
        }
        let row_count = self.row_count as usize;
        if top_k == 0 || row_count == 0 {
            return Ok(self.row_count);
        }
        let metric_count = self.metric_count as usize;
        let rank_values = match &self.metric_values {
            CpuPrecisionValues::F32(values) => CpuPrecisionValues::F32(
                (0..row_count)
                    .map(|row| values[row * metric_count + metric_index])
                    .collect(),
            ),
            CpuPrecisionValues::F64(values) => CpuPrecisionValues::F64(
                (0..row_count)
                    .map(|row| values[row * metric_count + metric_index])
                    .collect(),
            ),
        };
        let selected = match &rank_values {
            CpuPrecisionValues::F32(values) => {
                top_k_precision_indices(CpuPrecisionSlice::F32(values), top_k, descending)
            }
            CpuPrecisionValues::F64(values) => {
                top_k_precision_indices(CpuPrecisionSlice::F64(values), top_k, descending)
            }
        };
        self.retain_rows(&selected)?;
        Ok(self.row_count)
    }

    fn retain_rows(&mut self, rows: &[usize]) -> OrchestratorResult<()> {
        let max_arity = self.max_arity as usize;
        let metric_count = self.metric_count as usize;
        let source_combo = self.combo_indices.clone();
        let source_ranks = self.ranks.clone();
        let source_families = self.families.clone();
        let source_candidate_ids = self.candidate_ids.clone();
        let source_flags = self.row_flags.clone();
        match &mut self.metric_values {
            CpuPrecisionValues::F32(values) => {
                let source_values = values.clone();
                for (destination, &source) in rows.iter().enumerate() {
                    copy_structural_row(
                        &source_combo,
                        &source_ranks,
                        &source_families,
                        &source_candidate_ids,
                        &source_flags,
                        &mut self.combo_indices,
                        &mut self.ranks,
                        &mut self.families,
                        &mut self.candidate_ids,
                        &mut self.row_flags,
                        source,
                        destination,
                        max_arity,
                    )?;
                    let source_offset = source * metric_count;
                    let destination_offset = destination * metric_count;
                    values[destination_offset..destination_offset + metric_count].copy_from_slice(
                        &source_values[source_offset..source_offset + metric_count],
                    );
                }
            }
            CpuPrecisionValues::F64(values) => {
                let source_values = values.clone();
                for (destination, &source) in rows.iter().enumerate() {
                    copy_structural_row(
                        &source_combo,
                        &source_ranks,
                        &source_families,
                        &source_candidate_ids,
                        &source_flags,
                        &mut self.combo_indices,
                        &mut self.ranks,
                        &mut self.families,
                        &mut self.candidate_ids,
                        &mut self.row_flags,
                        source,
                        destination,
                        max_arity,
                    )?;
                    let source_offset = source * metric_count;
                    let destination_offset = destination * metric_count;
                    values[destination_offset..destination_offset + metric_count].copy_from_slice(
                        &source_values[source_offset..source_offset + metric_count],
                    );
                }
            }
        }
        self.row_count = rows.len() as u64;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_structural_row(
    source_combo: &[u32],
    _source_ranks: &[u32],
    source_families: &[u32],
    source_candidate_ids: &[u64],
    source_flags: &[u32],
    destination_combo: &mut [u32],
    destination_ranks: &mut [u32],
    destination_families: &mut [u32],
    destination_candidate_ids: &mut [u64],
    destination_flags: &mut [u32],
    source: usize,
    destination: usize,
    max_arity: usize,
) -> OrchestratorResult<()> {
    let source_offset = source
        .checked_mul(max_arity)
        .ok_or(OrchestratorError::InvalidPlan(
            "CPU precision result source combo offset overflows",
        ))?;
    let destination_offset =
        destination
            .checked_mul(max_arity)
            .ok_or(OrchestratorError::InvalidPlan(
                "CPU precision result destination combo offset overflows",
            ))?;
    destination_combo[destination_offset..destination_offset + max_arity]
        .copy_from_slice(&source_combo[source_offset..source_offset + max_arity]);
    destination_ranks[destination] = u32::try_from(destination)
        .map_err(|_| OrchestratorError::InvalidPlan("CPU precision result rank exceeds u32"))?;
    destination_families[destination] = source_families[source];
    destination_candidate_ids[destination] = source_candidate_ids[source];
    destination_flags[destination] = source_flags[source];
    Ok(())
}

#[derive(Debug)]
pub struct OwnedResultTable {
    raw: GafimeResultTable,
    combo_indices: Vec<u32>,
    metric_values: Vec<f32>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl OwnedResultTable {
    pub fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
        let mut table = Self {
            raw: GafimeResultTable {
                abi_version: GAFIME_ABI_VERSION,
                max_arity,
                metric_count,
                flags: 0,
                capacity,
                row_count: 0,
                combo_indices: core::ptr::null_mut(),
                metric_values: core::ptr::null_mut(),
                ranks: core::ptr::null_mut(),
                families: core::ptr::null_mut(),
                candidate_ids: core::ptr::null_mut(),
                row_flags: core::ptr::null_mut(),
                backend_private: core::ptr::null_mut(),
                reserved: [0; 8],
            },
            combo_indices: vec![u32::MAX; capacity as usize * max_arity as usize],
            metric_values: vec![0.0; capacity as usize * metric_count as usize],
            ranks: vec![0; capacity as usize],
            families: vec![0; capacity as usize],
            candidate_ids: vec![0; capacity as usize],
            row_flags: vec![0; capacity as usize],
        };
        table.rebind();
        table
    }

    pub fn raw(&self) -> &GafimeResultTable {
        &self.raw
    }

    pub fn raw_mut(&mut self) -> &mut GafimeResultTable {
        self.rebind();
        &mut self.raw
    }

    pub fn metric_values(&self) -> &[f32] {
        &self.metric_values
    }

    pub fn combo_indices(&self) -> &[u32] {
        &self.combo_indices
    }

    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    pub fn families(&self) -> &[u32] {
        &self.families
    }

    pub fn candidate_ids(&self) -> &[u64] {
        &self.candidate_ids
    }

    pub fn row_flags(&self) -> &[u32] {
        &self.row_flags
    }

    pub fn row_count(&self) -> usize {
        self.raw.row_count as usize
    }

    pub fn max_arity(&self) -> usize {
        self.raw.max_arity as usize
    }

    pub fn metric_count(&self) -> usize {
        self.raw.metric_count as usize
    }

    pub fn append_rows_from(
        &mut self,
        source: &Self,
        candidate_id_offset: u64,
    ) -> Result<(), &'static str> {
        if source.max_arity() > self.max_arity() {
            return Err("source result arity exceeds destination arity");
        }
        if source.metric_count() != self.metric_count() {
            return Err("source and destination metric widths differ");
        }
        let destination_max_arity = self.max_arity();
        let source_max_arity = source.max_arity();
        let metric_count = self.metric_count();
        let start = self.row_count();
        let end = start
            .checked_add(source.row_count())
            .ok_or("result row count overflows")?;
        if end > self.raw.capacity as usize {
            return Err("source rows exceed destination capacity");
        }

        for source_row in 0..source.row_count() {
            let destination_row = start + source_row;
            let destination_combo = destination_row * destination_max_arity;
            self.combo_indices[destination_combo..destination_combo + destination_max_arity]
                .fill(u32::MAX);
            let source_combo = source_row * source_max_arity;
            self.combo_indices[destination_combo..destination_combo + source_max_arity]
                .copy_from_slice(
                    &source.combo_indices[source_combo..source_combo + source_max_arity],
                );

            let destination_metrics = destination_row * metric_count;
            let source_metrics = source_row * metric_count;
            self.metric_values[destination_metrics..destination_metrics + metric_count]
                .copy_from_slice(
                    &source.metric_values[source_metrics..source_metrics + metric_count],
                );
            self.ranks[destination_row] = destination_row
                .try_into()
                .map_err(|_| "result rank exceeds u32")?;
            self.families[destination_row] = source.families[source_row];
            self.candidate_ids[destination_row] = source.candidate_ids[source_row]
                .checked_add(candidate_id_offset)
                .ok_or("candidate id overflows")?;
            self.row_flags[destination_row] = source.row_flags[source_row];
        }
        self.raw.flags |= source.raw.flags;
        self.raw.row_count = end as u64;
        Ok(())
    }

    pub fn with_raw_rows_mut<R>(
        &mut self,
        start: u64,
        capacity: u64,
        execute: impl FnOnce(&mut GafimeResultTable) -> R,
    ) -> Result<(R, u64), &'static str> {
        let start =
            usize::try_from(start).map_err(|_| "result row offset exceeds address space")?;
        let capacity =
            usize::try_from(capacity).map_err(|_| "result row capacity exceeds address space")?;
        let end = start
            .checked_add(capacity)
            .ok_or("result row window overflows")?;
        if end > self.raw.capacity as usize {
            return Err("result row window exceeds destination capacity");
        }
        let combo_offset = start
            .checked_mul(self.max_arity())
            .ok_or("result combo row window overflows")?;
        let metric_offset = start
            .checked_mul(self.metric_count())
            .ok_or("result metric row window overflows")?;
        let mut raw = self.raw;
        raw.capacity = capacity as u64;
        raw.row_count = 0;
        raw.combo_indices = self.combo_indices.as_mut_ptr().wrapping_add(combo_offset);
        raw.metric_values = self.metric_values.as_mut_ptr().wrapping_add(metric_offset);
        raw.ranks = self.ranks.as_mut_ptr().wrapping_add(start);
        raw.families = self.families.as_mut_ptr().wrapping_add(start);
        raw.candidate_ids = self.candidate_ids.as_mut_ptr().wrapping_add(start);
        raw.row_flags = self.row_flags.as_mut_ptr().wrapping_add(start);

        let value = execute(&mut raw);
        if raw.row_count > raw.capacity {
            return Err("backend exceeded result row window capacity");
        }
        self.raw.flags |= raw.flags;
        self.raw.backend_private = raw.backend_private;
        self.raw.reserved = raw.reserved;
        Ok((value, raw.row_count))
    }

    pub fn commit_appended_rows(
        &mut self,
        start: u64,
        row_count: u64,
        candidate_id_offset: u64,
    ) -> Result<(), &'static str> {
        if self.raw.row_count != start {
            return Err("appended result rows do not start at the current row count");
        }
        let start =
            usize::try_from(start).map_err(|_| "result row offset exceeds address space")?;
        let row_count =
            usize::try_from(row_count).map_err(|_| "result row count exceeds address space")?;
        let end = start
            .checked_add(row_count)
            .ok_or("appended result row count overflows")?;
        if end > self.raw.capacity as usize {
            return Err("appended result rows exceed destination capacity");
        }
        for row in start..end {
            self.ranks[row] = row.try_into().map_err(|_| "result rank exceeds u32")?;
            self.candidate_ids[row] = self.candidate_ids[row]
                .checked_add(candidate_id_offset)
                .ok_or("candidate id overflows")?;
        }
        self.raw.row_count = end as u64;
        Ok(())
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

/// Owned ABI 1.1 binary64 result table for the mixed and fp64 profiles.
///
/// Its metadata layout intentionally mirrors [`OwnedResultTable`], while the
/// only floating pointer is typed `*mut f64`.  This prevents an ABI caller from
/// reinterpreting a legacy `*mut f32` table as a double table.
#[derive(Debug)]
pub struct OwnedResultTableF64 {
    raw: GafimeResultTableF64,
    combo_indices: Vec<u32>,
    metric_values: Vec<f64>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl OwnedResultTableF64 {
    pub fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
        let mut table = Self {
            raw: GafimeResultTableF64 {
                abi_version: GAFIME_PRECISION_ABI_VERSION,
                max_arity,
                metric_count,
                flags: 0,
                capacity,
                row_count: 0,
                combo_indices: core::ptr::null_mut(),
                metric_values: core::ptr::null_mut(),
                ranks: core::ptr::null_mut(),
                families: core::ptr::null_mut(),
                candidate_ids: core::ptr::null_mut(),
                row_flags: core::ptr::null_mut(),
                backend_private: core::ptr::null_mut(),
                reserved: [0; 8],
            },
            combo_indices: vec![u32::MAX; capacity as usize * max_arity as usize],
            metric_values: vec![0.0; capacity as usize * metric_count as usize],
            ranks: vec![0; capacity as usize],
            families: vec![0; capacity as usize],
            candidate_ids: vec![0; capacity as usize],
            row_flags: vec![0; capacity as usize],
        };
        table.rebind();
        table
    }

    pub fn raw(&self) -> &GafimeResultTableF64 {
        &self.raw
    }

    pub fn raw_mut(&mut self) -> &mut GafimeResultTableF64 {
        self.rebind();
        &mut self.raw
    }

    pub fn metric_values(&self) -> &[f64] {
        &self.metric_values
    }

    pub fn metric_values_mut(&mut self) -> &mut [f64] {
        &mut self.metric_values
    }

    pub fn combo_indices(&self) -> &[u32] {
        &self.combo_indices
    }

    pub fn ranks(&self) -> &[u32] {
        &self.ranks
    }

    pub fn families(&self) -> &[u32] {
        &self.families
    }

    pub fn candidate_ids(&self) -> &[u64] {
        &self.candidate_ids
    }

    pub fn row_flags(&self) -> &[u32] {
        &self.row_flags
    }

    pub fn row_count(&self) -> usize {
        self.raw.row_count as usize
    }

    pub fn max_arity(&self) -> usize {
        self.raw.max_arity as usize
    }

    pub fn metric_count(&self) -> usize {
        self.raw.metric_count as usize
    }

    /// Append a typed f64 result table without changing any visible score bits.
    ///
    /// This is deliberately the same structural operation as
    /// [`OwnedResultTable::append_rows_from`], but it never passes the f64
    /// values through an f32 staging row.  Generated-family descriptor batches
    /// use it to concatenate unary and higher-order output while retaining the
    /// mixed/fp64 ranking lane.
    pub fn append_rows_from(
        &mut self,
        source: &Self,
        candidate_id_offset: u64,
    ) -> Result<(), &'static str> {
        if source.max_arity() > self.max_arity() {
            return Err("source result arity exceeds destination arity");
        }
        if source.metric_count() != self.metric_count() {
            return Err("source and destination metric widths differ");
        }
        let destination_max_arity = self.max_arity();
        let source_max_arity = source.max_arity();
        let metric_count = self.metric_count();
        let start = self.row_count();
        let end = start
            .checked_add(source.row_count())
            .ok_or("result row count overflows")?;
        if end > self.raw.capacity as usize {
            return Err("source rows exceed destination capacity");
        }

        for source_row in 0..source.row_count() {
            let destination_row = start + source_row;
            let destination_combo = destination_row * destination_max_arity;
            self.combo_indices[destination_combo..destination_combo + destination_max_arity]
                .fill(u32::MAX);
            let source_combo = source_row * source_max_arity;
            self.combo_indices[destination_combo..destination_combo + source_max_arity]
                .copy_from_slice(
                    &source.combo_indices[source_combo..source_combo + source_max_arity],
                );

            let destination_metrics = destination_row * metric_count;
            let source_metrics = source_row * metric_count;
            self.metric_values[destination_metrics..destination_metrics + metric_count]
                .copy_from_slice(
                    &source.metric_values[source_metrics..source_metrics + metric_count],
                );
            self.ranks[destination_row] = destination_row
                .try_into()
                .map_err(|_| "result rank exceeds u32")?;
            self.families[destination_row] = source.families[source_row];
            self.candidate_ids[destination_row] = source.candidate_ids[source_row]
                .checked_add(candidate_id_offset)
                .ok_or("candidate id overflows")?;
            self.row_flags[destination_row] = source.row_flags[source_row];
        }
        self.raw.flags |= source.raw.flags;
        self.raw.row_count = end as u64;
        Ok(())
    }

    /// Expose a bounded ABI-1.1 f64 output window for one synchronous backend
    /// launch.  The temporary descriptor is a view onto this owner's f64
    /// allocation; no legacy f32 ABI pointer is ever constructed.
    pub fn with_raw_rows_mut<R>(
        &mut self,
        start: u64,
        capacity: u64,
        execute: impl FnOnce(&mut GafimeResultTableF64) -> R,
    ) -> Result<(R, u64), &'static str> {
        let start =
            usize::try_from(start).map_err(|_| "result row offset exceeds address space")?;
        let capacity =
            usize::try_from(capacity).map_err(|_| "result row capacity exceeds address space")?;
        let end = start
            .checked_add(capacity)
            .ok_or("result row window overflows")?;
        if end > self.raw.capacity as usize {
            return Err("result row window exceeds destination capacity");
        }
        let combo_offset = start
            .checked_mul(self.max_arity())
            .ok_or("result combo row window overflows")?;
        let metric_offset = start
            .checked_mul(self.metric_count())
            .ok_or("result metric row window overflows")?;
        let mut raw = self.raw;
        raw.capacity = capacity as u64;
        raw.row_count = 0;
        raw.combo_indices = self.combo_indices.as_mut_ptr().wrapping_add(combo_offset);
        raw.metric_values = self.metric_values.as_mut_ptr().wrapping_add(metric_offset);
        raw.ranks = self.ranks.as_mut_ptr().wrapping_add(start);
        raw.families = self.families.as_mut_ptr().wrapping_add(start);
        raw.candidate_ids = self.candidate_ids.as_mut_ptr().wrapping_add(start);
        raw.row_flags = self.row_flags.as_mut_ptr().wrapping_add(start);

        let value = execute(&mut raw);
        if raw.row_count > raw.capacity {
            return Err("backend exceeded result row window capacity");
        }
        self.raw.flags |= raw.flags;
        self.raw.backend_private = raw.backend_private;
        self.raw.reserved = raw.reserved;
        Ok((value, raw.row_count))
    }

    /// Commit a row window after a backend wrote directly into it.
    ///
    /// Candidate identity and rank position are structural integers; the only
    /// floating data in this operation is the already-written f64 result row.
    pub fn commit_appended_rows(
        &mut self,
        start: u64,
        row_count: u64,
        candidate_id_offset: u64,
    ) -> Result<(), &'static str> {
        if self.raw.row_count != start {
            return Err("appended result rows do not start at the current row count");
        }
        let start =
            usize::try_from(start).map_err(|_| "result row offset exceeds address space")?;
        let row_count =
            usize::try_from(row_count).map_err(|_| "result row count exceeds address space")?;
        let end = start
            .checked_add(row_count)
            .ok_or("appended result row count overflows")?;
        if end > self.raw.capacity as usize {
            return Err("appended result rows exceed destination capacity");
        }
        for row in start..end {
            self.ranks[row] = row.try_into().map_err(|_| "result rank exceeds u32")?;
            self.candidate_ids[row] = self.candidate_ids[row]
                .checked_add(candidate_id_offset)
                .ok_or("candidate id overflows")?;
        }
        self.raw.row_count = end as u64;
        Ok(())
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

/// Typed result ownership selected once from the canonical public profile.
/// `Fp32` owns the legacy f32 ABI table; `Mixed` and `Fp64` own the additive
/// ABI 1.1 f64 table. There is no independently configurable result dtype.
#[derive(Debug)]
pub enum PrecisionOwnedResultTable {
    Fp32(OwnedResultTable),
    /// Mixed and fp64 use the same ABI-1.1 f64 layout, but they are distinct
    /// resident/artifact identities.  Keep the exact profile with the owner so
    /// a report cannot accidentally relabel a true fp64 run as mixed.
    F64 {
        profile: PrecisionProfile,
        table: OwnedResultTableF64,
    },
}

impl PrecisionOwnedResultTable {
    pub fn new(
        profile: PrecisionProfile,
        capacity: u64,
        max_arity: u32,
        metric_count: u32,
    ) -> Self {
        match profile.cpu_contract().ranking_and_public_result {
            CpuDtype::F32 => Self::Fp32(OwnedResultTable::new(capacity, max_arity, metric_count)),
            CpuDtype::F64 => Self::F64 {
                profile,
                table: OwnedResultTableF64::new(capacity, max_arity, metric_count),
            },
        }
    }

    /// Exact public profile that selected this typed result owner.
    pub fn profile(&self) -> PrecisionProfile {
        match self {
            Self::Fp32(_) => PrecisionProfile::Fp32,
            Self::F64 { profile, .. } => *profile,
        }
    }

    pub fn f32_mut(&mut self) -> Option<&mut GafimeResultTable> {
        match self {
            Self::Fp32(table) => Some(table.raw_mut()),
            Self::F64 { .. } => None,
        }
    }

    pub fn f64_mut(&mut self) -> Option<&mut GafimeResultTableF64> {
        match self {
            Self::Fp32(_) => None,
            Self::F64 { table, .. } => Some(table.raw_mut()),
        }
    }

    pub fn as_f32(&self) -> Option<&OwnedResultTable> {
        match self {
            Self::Fp32(table) => Some(table),
            Self::F64 { .. } => None,
        }
    }

    pub fn as_f64(&self) -> Option<&OwnedResultTableF64> {
        match self {
            Self::Fp32(_) => None,
            Self::F64 { table, .. } => Some(table),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_result_table_keeps_f64_scores_through_visible_ranking() {
        let mut table = CpuPrecisionResultTable::new(PrecisionProfile::Mixed, 3, 1, 1).unwrap();
        table
            .push_row(&[0], CpuPrecisionSlice::F64(&[0.5]), 1, 10, 0)
            .unwrap();
        table
            .push_row(
                &[1],
                CpuPrecisionSlice::F64(&[0.500_000_000_000_000_1]),
                1,
                11,
                0,
            )
            .unwrap();
        table
            .push_row(&[2], CpuPrecisionSlice::F64(&[0.25]), 1, 12, 0)
            .unwrap();
        table.compact_top_k(0, true, 2).unwrap();

        assert_eq!(table.candidate_ids(), &[11, 10]);
        let CpuPrecisionValues::F64(values) = table.metric_values() else {
            panic!("mixed public results must stay f64")
        };
        assert_eq!(values[0].to_bits(), 0.500_000_000_000_000_1f64.to_bits());
        assert_eq!(table.ranks(), &[0, 1]);
    }

    #[test]
    fn precision_result_compaction_drops_nonfinite_scores() {
        let mut table = CpuPrecisionResultTable::new(PrecisionProfile::Fp32, 3, 1, 1).unwrap();
        table
            .push_row(&[0], CpuPrecisionSlice::F32(&[f32::NAN]), 1, 10, 0)
            .unwrap();
        table
            .push_row(&[1], CpuPrecisionSlice::F32(&[0.5]), 1, 11, 0)
            .unwrap();
        table
            .push_row(&[2], CpuPrecisionSlice::F32(&[f32::INFINITY]), 1, 12, 0)
            .unwrap();

        assert_eq!(table.compact_top_k(0, true, 3).unwrap(), 1);
        assert_eq!(table.candidate_ids(), &[11]);
        assert_eq!(table.ranks(), &[0]);
    }

    #[test]
    fn precision_result_table_rejects_profile_result_downcast() {
        let mut table = CpuPrecisionResultTable::new(PrecisionProfile::Fp64, 1, 1, 1).unwrap();
        assert!(table
            .push_row(&[0], CpuPrecisionSlice::F32(&[1.0]), 1, 0, 0)
            .is_err());
        assert_eq!(table.row_count(), 0);
    }

    #[test]
    fn fp32_result_table_rejects_hidden_f64_ranking_values() {
        let mut table = CpuPrecisionResultTable::new(PrecisionProfile::Fp32, 1, 1, 1).unwrap();
        assert!(table
            .push_row(&[0], CpuPrecisionSlice::F64(&[1.0]), 1, 0, 0)
            .is_err());
    }

    #[test]
    fn owned_f64_result_table_exposes_only_typed_f64_abi_storage() {
        let mut table = OwnedResultTableF64::new(2, 1, 1);
        let raw = table.raw_mut();
        assert_eq!(raw.abi_version, GAFIME_PRECISION_ABI_VERSION);
        assert!(!raw.metric_values.is_null());
        // SAFETY: the owner allocated exactly two f64 metric slots and raw_mut
        // rebound its pointer to the same live allocation.
        unsafe {
            *raw.metric_values = 0.500_000_000_000_000_1;
        }
        assert_eq!(
            table.metric_values()[0].to_bits(),
            0.500_000_000_000_000_1f64.to_bits()
        );

        let mut mixed = PrecisionOwnedResultTable::new(PrecisionProfile::Mixed, 1, 1, 1);
        assert!(mixed.f32_mut().is_none());
        assert!(mixed.f64_mut().is_some());
        assert_eq!(mixed.profile(), PrecisionProfile::Mixed);
        let fp64 = PrecisionOwnedResultTable::new(PrecisionProfile::Fp64, 1, 1, 1);
        assert_eq!(fp64.profile(), PrecisionProfile::Fp64);
        let mut fp32 = PrecisionOwnedResultTable::new(PrecisionProfile::Fp32, 1, 1, 1);
        assert!(fp32.f32_mut().is_some());
        assert!(fp32.f64_mut().is_none());
    }

    #[test]
    fn f64_owner_appends_and_commits_windows_without_f32_conversion() {
        let visible = 0.500_000_000_000_000_1f64;
        let mut source = OwnedResultTableF64::new(1, 1, 1);
        source.combo_indices[0] = 9;
        source.metric_values[0] = visible;
        source.families[0] = 3;
        source.candidate_ids[0] = 2;
        source.raw.row_count = 1;

        let mut destination = OwnedResultTableF64::new(3, 2, 1);
        destination.append_rows_from(&source, 10).unwrap();
        assert_eq!(destination.row_count(), 1);
        assert_eq!(&destination.combo_indices[..2], &[9, u32::MAX]);
        assert_eq!(destination.metric_values[0].to_bits(), visible.to_bits());
        assert_eq!(destination.candidate_ids[0], 12);

        let ((), written) = destination
            .with_raw_rows_mut(1, 2, |raw| {
                // SAFETY: this f64 window has two owned rows and one metric
                // each; values are written directly through ABI 1.1 storage.
                unsafe {
                    *raw.combo_indices.add(0) = 4;
                    *raw.combo_indices.add(1) = 5;
                    *raw.metric_values.add(0) = visible;
                    *raw.metric_values.add(1) = 0.25;
                    *raw.candidate_ids.add(0) = 0;
                    *raw.candidate_ids.add(1) = 1;
                }
                raw.row_count = 2;
            })
            .unwrap();
        destination.commit_appended_rows(1, written, 20).unwrap();
        assert_eq!(destination.row_count(), 3);
        assert_eq!(destination.metric_values()[1].to_bits(), visible.to_bits());
        assert_eq!(&destination.candidate_ids()[..3], &[12, 20, 21]);
    }

    #[test]
    fn appending_rows_pads_combos_and_offsets_candidate_ids() {
        let mut source = OwnedResultTable::new(2, 1, 1);
        source.combo_indices[..2].copy_from_slice(&[4, 2]);
        source.metric_values[..2].copy_from_slice(&[0.5, 0.25]);
        source.families[..2].fill(1);
        source.candidate_ids[..2].copy_from_slice(&[0, 1]);
        source.raw.row_count = 2;

        let mut destination = OwnedResultTable::new(3, 2, 1);
        destination.append_rows_from(&source, 7).unwrap();

        assert_eq!(destination.row_count(), 2);
        assert_eq!(&destination.combo_indices[..4], &[4, u32::MAX, 2, u32::MAX]);
        assert_eq!(&destination.metric_values[..2], &[0.5, 0.25]);
        assert_eq!(&destination.candidate_ids[..2], &[7, 8]);
        assert_eq!(&destination.ranks[..2], &[0, 1]);
    }

    #[test]
    fn backend_can_write_directly_into_a_bounded_result_row_window() {
        let mut destination = OwnedResultTable::new(3, 2, 1);
        let mut source = OwnedResultTable::new(1, 1, 1);
        source.combo_indices[0] = 4;
        source.metric_values[0] = 0.75;
        source.candidate_ids[0] = 0;
        source.raw.row_count = 1;
        destination.append_rows_from(&source, 0).unwrap();

        let ((), written) = destination
            .with_raw_rows_mut(1, 2, |raw| {
                // SAFETY: the bounded row view owns capacity for two rows with
                // two combo slots and one metric per row.
                unsafe {
                    *raw.combo_indices.add(0) = 1;
                    *raw.combo_indices.add(1) = 2;
                    *raw.combo_indices.add(2) = 2;
                    *raw.combo_indices.add(3) = 3;
                    *raw.metric_values.add(0) = 0.5;
                    *raw.metric_values.add(1) = 0.25;
                    *raw.candidate_ids.add(0) = 0;
                    *raw.candidate_ids.add(1) = 1;
                }
                raw.row_count = 2;
            })
            .unwrap();
        destination.commit_appended_rows(1, written, 1).unwrap();

        assert_eq!(destination.row_count(), 3);
        assert_eq!(&destination.combo_indices[..6], &[4, u32::MAX, 1, 2, 2, 3]);
        assert_eq!(&destination.metric_values[..3], &[0.75, 0.5, 0.25]);
        assert_eq!(&destination.candidate_ids[..3], &[0, 1, 2]);
        assert_eq!(&destination.ranks[..3], &[0, 1, 2]);
    }
}
