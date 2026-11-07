//! **Start the servers a file names and print what a task would be offered** — and, given a tool
//! and its arguments, call it once.
//!
//!     cargo run -p argmind-env-mcp --example try-servers -- %LOCALAPPDATA%\argmind\mcp.json
//!     cargo run -p argmind-env-mcp --example try-servers -- mcp.json everything__echo "{\"message\":\"hi\"}"

use argmind_core::task::Tools;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("a mcp.json");
    let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
    let (list, bad) = argmind_env_mcp::configured(&file);
    for b in bad {
        println!("  skipped: {b}");
    }
    let began = std::time::Instant::now();
    let (mut set, failed) = argmind_env_mcp::Servers::start(&list);
    println!("  started {} of {} in {} ms", set.servers.len(), list.len(), began.elapsed().as_millis());
    for f in failed {
        println!("  failed: {f}");
    }
    for s in &set.servers {
        if s.dropped > 0 {
            println!("  {}: {} more tools not offered", s.name, s.dropped);
        }
    }
    for t in set.list() {
        println!("- {} {} — {}", t.name, t.args, t.does);
    }
    if let (Some(tool), Some(args)) = (a.next(), a.next()) {
        let args: serde_json::Value = serde_json::from_str(&args).expect("args json");
        let began = std::time::Instant::now();
        let got = set.call(&tool, &args);
        println!("  {tool} in {} ms: {got:?}", began.elapsed().as_millis());
    }
    let began = std::time::Instant::now();
    drop(set);
    println!("  stopped in {} ms", began.elapsed().as_millis());
}
