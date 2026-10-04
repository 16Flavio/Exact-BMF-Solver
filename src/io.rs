//! Reading and writing binary matrices.
//!
//! Rows are either packed digits (`0111`) or space separated values, with an
//! optional `m n` header. Blank lines and lines starting with `#` are skipped.
//! A missing entry (`?` or `nan`) is excluded from the error.

use crate::bits::BitVec;
use std::fs;
use std::path::Path;

pub struct RawMat {
    pub name: String,
    pub m: usize,
    pub n: usize,
    /// Rows of X. A missing entry is stored as zero; see `known`.
    pub rows: Vec<BitVec>,
    /// Mask of known entries, all ones for a complete matrix.
    pub known: Vec<BitVec>,
}

impl RawMat {
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.rows[i].get(j)
    }

    pub fn is_known(&self, i: usize, j: usize) -> bool {
        self.known[i].get(j)
    }

    pub fn ones(&self) -> u64 {
        self.rows.iter().map(|r| r.count() as u64).sum()
    }

    pub fn missing(&self) -> u64 {
        (self.m * self.n) as u64 - self.known.iter().map(|r| r.count() as u64).sum::<u64>()
    }
}

/// Parses one row in either format. `None` is a missing entry.
fn parse_values(line: &str, no: usize) -> Result<Vec<Option<bool>>, String> {
    let mut v = Vec::new();
    if line.split_ascii_whitespace().count() > 1 || line.trim().len() == 1 {
        for t in line.split_ascii_whitespace() {
            // Float spellings (`1.0`, `nan`) are accepted for numpy output.
            match t.to_ascii_lowercase().as_str() {
                "0" | "0.0" | "0." => v.push(Some(false)),
                "1" | "1.0" | "1." => v.push(Some(true)),
                "?" | "nan" | "na" | "-" => v.push(None),
                _ => return Err(format!("line {} : non binary value '{}'", no, t)),
            }
        }
    } else {
        for c in line.trim().chars() {
            match c {
                '0' => v.push(Some(false)),
                '1' => v.push(Some(true)),
                '?' => v.push(None),
                _ => return Err(format!("line {} : non binary character '{}'", no, c)),
            }
        }
    }
    Ok(v)
}

pub fn load(path: &Path) -> Result<RawMat, String> {
    let txt = fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let lines: Vec<(usize, &str)> = txt
        .lines()
        .enumerate()
        .map(|(k, l)| (k + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if lines.is_empty() {
        return Err(format!("{}: empty file", path.display()));
    }

    // A first line "m n" is a header only if the line count matches, so that
    // a two column matrix starting with "1 0" is not mistaken for one.
    let header: Vec<&str> = lines[0].1.split_ascii_whitespace().collect();
    let mut start = 0;
    let mut expected: Option<(usize, usize)> = None;
    if header.len() == 2 {
        if let (Ok(a), Ok(b)) = (header[0].parse::<usize>(), header[1].parse::<usize>()) {
            if (a > 1 || b > 1) && lines.len() == a + 1 {
                start = 1;
                expected = Some((a, b));
            }
        }
    }

    let mut rows_bool = Vec::new();
    for &(no, l) in &lines[start..] {
        rows_bool.push(parse_values(l, no)?);
    }
    let n = rows_bool[0].len();
    if n == 0 {
        return Err(format!("{}: first line with no value", path.display()));
    }
    for (k, r) in rows_bool.iter().enumerate() {
        if r.len() != n {
            return Err(format!(
                "{}: line {} has {} values, {} expected",
                path.display(),
                k + 1,
                r.len(),
                n
            ));
        }
    }
    let m = rows_bool.len();
    if let Some((a, b)) = expected {
        if (a, b) != (m, n) {
            return Err(format!(
                "{}: header {}x{} but {}x{} read",
                path.display(),
                a,
                b,
                m,
                n
            ));
        }
    }

    let mut rows = vec![BitVec::zeros(n); m];
    let mut known = vec![BitVec::zeros(n); m];
    for i in 0..m {
        for j in 0..n {
            if let Some(b) = rows_bool[i][j] {
                known[i].set(j, true);
                if b {
                    rows[i].set(j, true);
                }
            }
        }
    }
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(RawMat {
        name,
        m,
        n,
        rows,
        known,
    })
}

/// Writes W and H in the dense format, with the dimension headers.
pub fn dump_factors(w: &[Vec<bool>], h: &[Vec<bool>]) -> String {
    let (m, r, n) = (w.len(), h.len(), h[0].len());
    let mut s = String::new();
    s.push_str(&format!("W {} {}\n", m, r));
    for line in w {
        let t: Vec<&str> = line.iter().map(|&b| if b { "1" } else { "0" }).collect();
        s.push_str(&t.join(" "));
        s.push('\n');
    }
    s.push_str(&format!("H {} {}\n", r, n));
    for line in h {
        let t: Vec<&str> = line.iter().map(|&b| if b { "1" } else { "0" }).collect();
        s.push_str(&t.join(" "));
        s.push('\n');
    }
    s
}

/// Reads W and H in the format written by `dump_factors`.
pub fn load_factors(path: &Path) -> Result<(Vec<Vec<bool>>, Vec<Vec<bool>>), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let mut block = |tag: &str| -> Result<Vec<Vec<bool>>, String> {
        let header = lines.next().ok_or(format!("missing {tag} header"))?;
        let dims: Vec<&str> = header.split_ascii_whitespace().collect();
        if dims.len() != 3 || dims[0] != tag {
            return Err(format!("expected '{tag} rows cols', got '{header}'"));
        }
        let rows: usize = dims[1].parse().map_err(|_| format!("bad header '{header}'"))?;
        let cols: usize = dims[2].parse().map_err(|_| format!("bad header '{header}'"))?;
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let line = lines.next().ok_or(format!("{tag}: missing row"))?;
            let row: Vec<bool> = line
                .split_ascii_whitespace()
                .map(|t| match t {
                    "0" => Ok(false),
                    "1" => Ok(true),
                    _ => Err(format!("{tag}: non binary value '{t}'")),
                })
                .collect::<Result<_, _>>()?;
            if row.len() != cols {
                return Err(format!("{tag}: row of {} values, expected {cols}", row.len()));
            }
            out.push(row);
        }
        Ok(out)
    };
    let w = block("W")?;
    let h = block("H")?;
    Ok((w, h))
}
