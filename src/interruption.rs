//! Graceful Ctrl+C handling.
//!
//! The first Ctrl+C sets a flag that the search loops poll, so the solver can
//! still report its `[LB, UB]` bracket. A second one exits immediately. The
//! system calls are declared by hand to avoid any dependency.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

static STOP: AtomicBool = AtomicBool::new(false);
static PRESSES: AtomicUsize = AtomicUsize::new(0);

fn received() {
    STOP.store(true, Ordering::Relaxed);
    if PRESSES.fetch_add(1, Ordering::Relaxed) >= 1 {
        std::process::exit(130);
    }
}

#[cfg(windows)]
mod platform {
    type Handler = unsafe extern "system" fn(u32) -> i32;

    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<Handler>, add: i32) -> i32;
    }

    // Events 0 and 1 are CTRL_C_EVENT and CTRL_BREAK_EVENT. Returning 1 marks
    // the event as handled, so Windows does not kill the process.
    unsafe extern "system" fn on_event(event: u32) -> i32 {
        if event <= 1 {
            super::received();
            1
        } else {
            0
        }
    }

    pub fn install() {
        unsafe { SetConsoleCtrlHandler(Some(on_event), 1) };
    }
}

#[cfg(not(windows))]
mod platform {
    extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }

    extern "C" fn on_event(_: i32) {
        super::received();
    }

    pub fn install() {
        // SIGINT is 2.
        unsafe { signal(2, on_event as *const () as usize) };
    }
}

/// Installs the handler. Call once at startup.
pub fn install() {
    platform::install();
}

/// True once the user has pressed Ctrl+C.
pub fn requested() -> bool {
    STOP.load(Ordering::Relaxed)
}

/// True if the deadline has passed or a stop was requested.
#[inline]
pub fn expired(deadline: Instant) -> bool {
    requested() || Instant::now() >= deadline
}

/// A deadline far in the future, used for "no time limit".
pub fn unlimited() -> Instant {
    Instant::now() + std::time::Duration::from_secs(10 * 365 * 24 * 3600)
}
