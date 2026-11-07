//! Against `examples/mcp-echo.rs`, which `cargo test` builds beside this binary: started, listed
//! across two pages, called past a notification and a request of the server's own, a failure read
//! as one, and stopped when the set is dropped.

use argmind_core::task::Tools;
use argmind_env_mcp::{configured, Servers};
use serde_json::json;

fn echo() -> String {
    let me = std::env::current_exe().unwrap();
    let dir = me.parent().unwrap().parent().unwrap().join("examples");
    let exe = dir.join(if cfg!(windows) { "mcp-echo.exe" } else { "mcp-echo" });
    assert!(exe.exists(), "{} — built by cargo test", exe.display());
    exe.display().to_string()
}

#[test]
fn a_configured_server_is_started_listed_called_and_stopped() {
    let file = json!({"mcpServers": {
        "my-echo": {"command": echo()},
        "off": {"command": "nothing", "disabled": true},
        "broken": {"args": ["x"]},
    }});
    let (list, bad) = configured(&file);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "my_echo");
    assert_eq!(bad, ["broken: command 가 없다"]);

    let (mut set, failed) = Servers::start(&list);
    assert!(failed.is_empty(), "{failed:?}");
    let specs = set.list();
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["my_echo__add", "my_echo__fail"], "both pages");
    //: The example holds the required arguments only, and the description says the optional one.
    assert_eq!(specs[0].args, json!({"a": 1, "b": 1}));
    assert!(specs[0].does.contains("note: string, 생략 가능"), "{}", specs[0].does);

    assert_eq!(set.call("my_echo__add", &json!({"a": 2, "b": 40})), Ok("42".into()));
    assert_eq!(set.call("my_echo__fail", &json!({})), Err("일부러 실패".into()));
    assert!(set.call("my_echo__nothing", &json!({})).is_err());
    assert!(set.call("other__add", &json!({})).is_err());
    drop(set);
}

#[test]
fn a_server_that_is_not_there_is_a_sentence_and_not_a_panic() {
    let (list, _) = configured(&json!({"mcpServers": {"x": {"command": "argmind-no-such-program-anywhere"}}}));
    let (set, failed) = Servers::start(&list);
    assert!(set.servers.is_empty());
    assert!(failed[0].starts_with("x: 시작하지 못했다"), "{failed:?}");
}

#[test]
fn none_configured_is_nothing() {
    let (list, bad) = configured(&json!({}));
    assert!(list.is_empty() && bad.is_empty());
}
