#[derive(Debug, Default)]
pub struct ScratchArena {
    f32_scratch: Vec<f32>,
}

impl ScratchArena {
    pub fn reserve_f32(&mut self, len: usize) -> &mut [f32] {
        self.f32_scratch.resize(len, 0.0);
        &mut self.f32_scratch
    }

    pub fn clear(&mut self) {
        self.f32_scratch.clear();
    }
}
