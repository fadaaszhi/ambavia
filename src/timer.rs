use std::time::{Duration, Instant};

use num_integer::Roots;

enum TimerAction {
    Start(String, Instant),
    Stop(Instant),
}

#[derive(Default)]
pub struct Timer {
    actions: Vec<TimerAction>,
    starts_minus_stops: usize,
}

impl Timer {
    pub fn start(&mut self, name: impl Into<String>) {
        self.actions
            .push(TimerAction::Start(name.into(), Instant::now()));
        self.starts_minus_stops += 1;
    }

    pub fn stop(&mut self) {
        self.actions.push(TimerAction::Stop(Instant::now()));

        if self.starts_minus_stops == 0 {
            panic!("stopped too many times");
        }

        self.starts_minus_stops -= 1;
    }

    pub fn stop_start(&mut self, name: &str) {
        self.stop();
        self.start(name);
    }

    pub fn string(&self) -> String {
        let mut stack = vec![];
        let mut times = vec![];
        let mut accounted_times = vec![];

        for action in &self.actions {
            match action {
                TimerAction::Start(_, start) => {
                    stack.push((times.len(), start));
                    times.push(Default::default());
                    accounted_times.push(None);
                }
                TimerAction::Stop(stop) => {
                    let (i, start) = stack.pop().unwrap();
                    times[i] = stop.duration_since(*start);

                    if let Some((j, _)) = stack.last() {
                        if let Some(accounted_time) = accounted_times.get_mut(*j) {
                            *accounted_time = Some(times[i] + accounted_time.unwrap_or_default());
                        }
                    }
                }
            }
        }

        let mut start = false;
        let mut stop = false;
        let mut indents = 0usize;
        let mut result = vec![];
        let mut times = times.into_iter();
        let mut accounted_times = accounted_times.into_iter();
        use std::io::Write;

        for action in &self.actions {
            match action {
                TimerAction::Start(name, _) => {
                    if start {
                        write!(result, " {{").unwrap();
                        indents += 1;
                    }

                    start = true;
                    stop = false;
                    write!(result, "\n{}", "  ".repeat(indents)).unwrap();
                    let time = times.next().unwrap();
                    write!(result, "{name}: {time:?}").unwrap();

                    if let Some(accounted_time) = accounted_times.next().unwrap() {
                        write!(result, " ({:?} unaccounted)", time - accounted_time).unwrap();
                    }
                }
                TimerAction::Stop(_) => {
                    if stop {
                        indents -= 1;
                        write!(result, "\n{}", "  ".repeat(indents)).unwrap();
                        write!(result, "}}").unwrap();
                    }

                    start = false;
                    stop = true;
                }
            }
        }

        String::from_utf8(result).unwrap()
    }
}

pub struct DurationStatsTracker {
    count: usize,
    total: Duration,
    total_of_squared: u128,
    min: Duration,
    max: Duration,
}

impl Default for DurationStatsTracker {
    fn default() -> Self {
        Self {
            count: 0,
            total: Duration::ZERO,
            total_of_squared: 0,
            min: Duration::from_secs(u64::MAX),
            max: Duration::from_secs(u64::MIN),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DurationStats {
    pub count: usize,
    pub total: Duration,
    pub mean: Duration,
    pub stdev: Duration,
    pub min: Duration,
    pub max: Duration,
}

impl DurationStatsTracker {
    pub fn track(&mut self, duration: Duration) {
        self.count += 1;
        self.total += duration;
        self.total_of_squared += duration.as_nanos().pow(2);
        self.min = self.min.min(duration);
        self.max = self.max.max(duration);
    }

    pub fn get_stats(&self) -> DurationStats {
        DurationStats {
            count: self.count,
            total: self.total,
            mean: self.total / self.count as u32,
            stdev: Duration::from_nanos(
                ((self.total_of_squared - self.total.as_nanos().pow(2) / self.count.max(1) as u128)
                    / (self.count - 1).max(1) as u128)
                    .sqrt() as _,
            ),
            min: self.min,
            max: self.max,
        }
    }
}
