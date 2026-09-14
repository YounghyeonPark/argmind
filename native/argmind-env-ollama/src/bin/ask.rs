//! One live call. The only part of the transport a test cannot reach is the socket, and this is it.
use argmind_env_ollama::Local;

fn main() {
    let prompt = std::env::args().nth(1)
        .unwrap_or_else(|| "Reply with only JSON: {\"say\": \"ready\"}".to_string());
    let mut local = Local::new("qwen3:8b");
    local.think = Some(false);
    let t0 = std::time::Instant::now();
    match local.ask(&prompt) {
        Ok(r) => {
            println!("  {:.0} ms   reason {:?}   prompt_tokens {:?}   out_tokens {:?}   thought {}",
                     t0.elapsed().as_secs_f64() * 1000.0, r.reason, r.prompt_tokens, r.out_tokens,
                     r.thought);
            println!("  text: {:?}", r.text);
            println!("  json: {:?}", argmind_core::llm::first_object(&r.text));
        }
        Err(e) => println!("  the call failed: {e}"),
    }
}
