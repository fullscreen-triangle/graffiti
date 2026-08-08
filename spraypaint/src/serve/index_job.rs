//! The background index build.
//!
//! `build_index` takes a *blocking exclusive* lock and can run for a long time
//! on a large repo. Running it on a request thread would occupy one of four
//! workers for the whole build, and the HTTP client would see an opaque stall
//! with no way to tell "working" from "hung". So the build runs on its own
//! thread, the request returns 202 immediately, and progress is polled from
//! `GET /api/index/status`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::actions;
use crate::config::SprayConfig;

/// What a build is currently doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// No build has been started in this server's lifetime.
    Idle,
    Running,
    /// The last build finished. Carries a human summary.
    Done(String),
    /// The last build failed. Carries the error text.
    Failed(String),
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Idle => "idle",
            JobState::Running => "running",
            JobState::Done(_) => "done",
            JobState::Failed(_) => "failed",
        }
    }
    pub fn detail(&self) -> Option<&str> {
        match self {
            JobState::Done(s) | JobState::Failed(s) => Some(s),
            _ => None,
        }
    }
}

/// A single-slot index build.
///
/// `running` is a separate atomic rather than being derived from `state` so the
/// "is one already going?" test is a lock-free load. A status poll must never
/// contend with the thread that is writing the result.
pub struct IndexJob {
    running: Arc<AtomicBool>,
    state: Arc<Mutex<JobState>>,
}

impl IndexJob {
    pub fn new() -> Self {
        IndexJob {
            running: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(JobState::Idle)),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn state(&self) -> JobState {
        self.state.lock().map(|g| g.clone()).unwrap_or(JobState::Idle)
    }

    /// Start a build unless one is already running.
    ///
    /// Returns `false` if a build was already in flight. The check is a
    /// `compare_exchange`, not a load-then-store, so two POSTs arriving on two
    /// worker threads at once cannot both win — one gets 202, the other 409.
    pub fn start(&self, root: PathBuf, cfg: SprayConfig) -> bool {
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }

        if let Ok(mut g) = self.state.lock() {
            *g = JobState::Running;
        }

        let running = Arc::clone(&self.running);
        let state = Arc::clone(&self.state);
        std::thread::spawn(move || {
            let result = actions::build_index(&root, &cfg);
            if let Ok(mut g) = state.lock() {
                *g = match result {
                    Ok(s) => JobState::Done(format!(
                        "indexed {} document(s), {} passage(s), {} scene(s); fingerprint {}",
                        s.documents,
                        s.passages,
                        s.scenes,
                        &s.identity_fingerprint[..s.identity_fingerprint.len().min(14)]
                    )),
                    // The error is reported, not swallowed and not panicked on:
                    // a failed build must leave the server answering requests.
                    Err(e) => JobState::Failed(format!("{e:#}")),
                };
            }
            // Released last, so a status poll that observes `running == false`
            // is guaranteed to read the final state, not the stale one.
            running.store(false, Ordering::SeqCst);
        });
        true
    }
}

impl Default for IndexJob {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_job_is_idle_and_not_running() {
        let j = IndexJob::new();
        assert!(!j.is_running());
        assert_eq!(j.state(), JobState::Idle);
    }

    #[test]
    fn state_strings_are_stable() {
        // The UI switches on these; renaming one silently breaks it.
        assert_eq!(JobState::Idle.as_str(), "idle");
        assert_eq!(JobState::Running.as_str(), "running");
        assert_eq!(JobState::Done(String::new()).as_str(), "done");
        assert_eq!(JobState::Failed(String::new()).as_str(), "failed");
    }

    /// Only one build may be in flight. Without the compare_exchange, two
    /// concurrent POSTs would both spawn a builder and the second would block
    /// on the exclusive lock the first holds.
    #[test]
    fn a_second_start_is_refused_while_one_is_running() {
        let j = IndexJob::new();
        j.running.store(true, Ordering::SeqCst);
        assert!(!j.start(PathBuf::from("."), SprayConfig::default()));
    }
}
