//! Conditional-route fidelity diagnostics. Never selected by the server.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::PathBuf, rc::Rc};

#[derive(Deserialize)]
struct Route {
    logits: Vec<f32>,
    scores: Vec<f32>,
    corrected: Vec<f32>,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

fn exact(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A recorded sorted top-k must contain each strictly greater score, no lower
/// score, and no duplicate/out-of-range index. Equal cutoff choices are legal.
fn legal(scores: &[f32], indices: &[usize], n: usize) -> Result<()> {
    ensure!(
        n > 0 && n <= scores.len() && indices.len() == n,
        "invalid top-k length"
    );
    ensure!(
        scores.iter().all(|x| x.is_finite()),
        "nonfinite router scores"
    );
    let selected: BTreeSet<_> = indices.iter().copied().collect();
    ensure!(
        selected.len() == n && indices.iter().all(|&i| i < scores.len()),
        "duplicate/out-of-range expert"
    );
    ensure!(
        indices.windows(2).all(|p| scores[p[0]] >= scores[p[1]]),
        "unsorted recorded top-k"
    );
    let cutoff = scores[*indices.last().unwrap()];
    ensure!(
        scores
            .iter()
            .enumerate()
            .all(|(i, &s)| s <= cutoff || selected.contains(&i)),
        "recording excludes strictly greater expert"
    );
    Ok(())
}

pub struct Replay {
    directory: PathBuf,
    pub records: Vec<Value>,
}

impl Replay {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            records: vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &mut self,
        api: &Rc<crate::aten::Api>,
        position: usize,
        layer: usize,
        logits: &[f32],
        bias: &[f32],
        n: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        let name = format!("p{position:03}-l{layer:02}-route.json");
        let expected: Route = serde_json::from_slice(&std::fs::read(self.directory.join(&name))?)?;
        ensure!(
            exact(logits, &expected.logits),
            "{name}: router logits differ before tie selection"
        );
        ensure!(
            expected.scores.len() == logits.len()
                && expected.corrected.len() == logits.len()
                && expected.weights.len() == n,
            "{name}: invalid recorded dimensions"
        );
        legal(&expected.corrected, &expected.indices, n).with_context(|| name.clone())?;
        let mut first_indices = vec![];
        for attempt in 1..=1024 {
            // Same production equations and ATen top-k; resampling chooses a
            // documented nondeterministic branch, never supplied indices/weights.
            let values = api.vectors(
                14,
                &[logits, bias],
                &[n as f64, scale as f64],
                2 * logits.len() + 2 * n,
            )?;
            let (scores, tail) = values.split_at(logits.len());
            let (corrected, routed) = tail.split_at(logits.len());
            ensure!(
                exact(scores, &expected.scores) && exact(corrected, &expected.corrected),
                "{name}: scores differ; replay refuses to conceal arithmetic divergence"
            );
            let ids: Vec<_> = routed[..n].iter().map(|&i| i as usize).collect();
            legal(corrected, &ids, n)?;
            if attempt == 1 {
                first_indices = ids.clone();
            }
            if ids == expected.indices {
                ensure!(
                    exact(&routed[n..], &expected.weights),
                    "{name}: same-branch weights differ"
                );
                self.records.push(json!({"position":position,"layer":layer,"attempts":attempt,
                    "first_indices":first_indices,"recorded_indices":ids,
                    "cutoff":corrected[ids[n-1]],"cutoff_ties":corrected.iter().enumerate()
                        .filter(|(_, s)| **s == corrected[ids[n-1]]).map(|(i,_)|i).collect::<Vec<_>>(),
                    "logits_scores_and_weights_exact":true}));
                return Ok(routed.to_vec());
            }
            ensure!(
                ids.iter()
                    .zip(&expected.indices)
                    .all(|(&a, &b)| corrected[a] == corrected[b]),
                "{name}: expert choice differs outside exact score ties"
            );
        }
        anyhow::bail!(
            "{name}: unchanged top-k did not reproduce recorded tied branch in 1024 attempts"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_only_legal_sorted_topk_with_exact_ties() {
        let s = [4., 3., 3., 1.];
        assert!(legal(&s, &[0, 1], 2).is_ok());
        assert!(legal(&s, &[0, 2], 2).is_ok());
        for ids in [[1, 2], [0, 3], [0, 0], [1, 0], [0, 4]] {
            assert!(legal(&s, &ids, 2).is_err(), "{ids:?}");
        }
        assert!(legal(&[f32::NAN], &[0], 1).is_err());
        assert!(legal(&s, &[], 0).is_err());
        assert!(!exact(&[0.], &[-0.]));
        assert!(!exact(&[1.], &[f32::from_bits(1f32.to_bits() + 1)]));
    }
}
