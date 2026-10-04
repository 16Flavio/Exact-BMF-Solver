//! Packed bit vectors, 64 entries per `u64`.
//!
//! Invariant: the bits of the last word past `len` are always zero. Every
//! counting operation relies on this.

pub type Word = u64;
pub const WBITS: usize = 64;

#[inline]
pub fn nwords(len: usize) -> usize {
    (len + WBITS - 1) / WBITS
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BitVec {
    len: usize,
    w: Vec<Word>,
}

impl BitVec {
    pub fn zeros(len: usize) -> Self {
        BitVec { len, w: vec![0; nwords(len)] }
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.w[i / WBITS] >> (i % WBITS)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        let (q, b) = (i / WBITS, i % WBITS);
        if v {
            self.w[q] |= 1 << b;
        } else {
            self.w[q] &= !(1 << b);
        }
    }

    /// The underlying words.
    #[inline]
    pub fn words(&self) -> &[Word] {
        &self.w
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn fill_zero(&mut self) {
        for x in self.w.iter_mut() {
            *x = 0;
        }
    }

    #[inline]
    pub fn copy_from(&mut self, o: &BitVec) {
        self.w.copy_from_slice(&o.w);
    }

    #[inline]
    pub fn or_in(&mut self, o: &BitVec) {
        for (a, b) in self.w.iter_mut().zip(o.w.iter()) {
            *a |= *b;
        }
    }

    /// Number of set bits.
    #[inline]
    pub fn count(&self) -> u32 {
        self.w.iter().map(|x| x.count_ones()).sum()
    }

    /// Hamming distance.
    #[inline]
    pub fn xor_count(&self, o: &BitVec) -> u32 {
        self.w.iter().zip(o.w.iter()).map(|(a, b)| (a ^ b).count_ones()).sum()
    }

    #[inline]
    pub fn and_in(&mut self, o: &BitVec) {
        for (a, b) in self.w.iter_mut().zip(o.w.iter()) {
            *a &= *b;
        }
    }

    /// True if every bit set in `self` is also set in `o`.
    #[inline]
    pub fn is_subset_of(&self, o: &BitVec) -> bool {
        self.w.iter().zip(o.w.iter()).all(|(a, b)| a & !b == 0)
    }

    /// Sets `self` to `a & !b`.
    #[inline]
    pub fn and_not_from(&mut self, a: &BitVec, b: &BitVec) {
        for (t, (x, y)) in self.w.iter_mut().zip(a.w.iter().zip(b.w.iter())) {
            *t = x & !y;
        }
    }

    /// Number of bits set in both.
    #[inline]
    pub fn and_count(&self, o: &BitVec) -> u32 {
        self.w.iter().zip(o.w.iter()).map(|(a, b)| (a & b).count_ones()).sum()
    }

    /// Number of bits set in `self` but not in `o`.
    #[inline]
    pub fn andnot_count(&self, o: &BitVec) -> u32 {
        self.w.iter().zip(o.w.iter()).map(|(a, b)| (a & !b).count_ones()).sum()
    }

    // Masked variants: only positions set in `m` are counted. They cost one
    // extra AND per word, so callers use them only when entries are missing.

    /// Hamming distance restricted to the mask.
    #[inline]
    pub fn xor_count_masked(&self, o: &BitVec, m: &BitVec) -> u32 {
        self.w
            .iter()
            .zip(o.w.iter())
            .zip(m.w.iter())
            .map(|((a, b), k)| ((a ^ b) & k).count_ones())
            .sum()
    }

    /// `and_count` restricted to the mask.
    #[inline]
    pub fn and_count_masked(&self, o: &BitVec, m: &BitVec) -> u32 {
        self.w
            .iter()
            .zip(o.w.iter())
            .zip(m.w.iter())
            .map(|((a, b), k)| (a & b & k).count_ones())
            .sum()
    }

    /// `andnot_count` restricted to the mask.
    #[inline]
    pub fn andnot_count_masked(&self, o: &BitVec, m: &BitVec) -> u32 {
        self.w
            .iter()
            .zip(o.w.iter())
            .zip(m.w.iter())
            .map(|((a, b), k)| (a & !b & k).count_ones())
            .sum()
    }

    /// `is_subset_of` restricted to the mask: `self` may cover a missing entry
    /// of `o`, since a missing entry costs nothing.
    #[inline]
    pub fn is_subset_of_masked(&self, o: &BitVec, m: &BitVec) -> bool {
        self.w
            .iter()
            .zip(o.w.iter())
            .zip(m.w.iter())
            .all(|((a, b), k)| a & !b & k == 0)
    }
}
