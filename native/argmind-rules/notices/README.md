# Licence texts the crates' own packages leave out

`examples/notices.rs` writes `THIRD-PARTY-NOTICES.txt` from the licence files each crate ships.
Three crates in the release graph ship none, all from one repository, and one of them carries
Microsoft code, so these three files were fetched once, on 2026-11-17, and are kept here verbatim:

| file | what it is | where it came from |
|---|---|---|
| `webview2-rs-LICENSE.txt` | the MIT licence of `webview2-com`, `webview2-com-sys` and `webview2-com-macros` | `https://raw.githubusercontent.com/wravery/webview2-rs/edc2caf886175ccaebe86078c9cfe1ae2a187328/LICENSE`, the commit the first two packages' `.cargo_vcs_info.json` name; the macros' commit, `dffa41a`, has the same file byte for byte |
| `webview2-sdk-LICENSE.txt` | the licence of Microsoft's WebView2 SDK, whose `WebView2LoaderStatic.lib` is linked into `argmind-app.exe` | `LICENSE.txt` inside `https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.3800.47` |
| `webview2-sdk-NOTICE.txt` | that SDK's third-party notice | `NOTICE.txt` in the same package |

**The SDK version comes from the loader itself.** `webview2-com-sys` 0.39.1's changelog says only
*update windows and WebView2 SDK versions*. The version was read from `x64/WebView2Loader.dll`'s
version resource, `1.0.3800.47`, in the crate's own package. `tests/third_party_notices.rs` fails
when `webview2-com-sys` is no longer 0.39.1, because a new one may carry a new loader.
