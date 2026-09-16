# A better voice — what exists, what it costs, 2026-09-16

> 목소리는 어색함.

The owner's verdict on `Microsoft Heami Desktop`, which is the only Korean voice this machine
has and is a Windows 7-era concatenative one. This is what was found looking for a better one,
including the two places the first answer was wrong.

## The rule that decides it

> 딥러닝 프레임워크도 가능함. 단, cpp 또는 rust를 사용해야 하고, 기존 프레임워크가 python이면
> cpp또는 rust로 자체 작성할것 — 2026-09-16

Plus the standing licence rule: only things with no open-source licence problem, held by
`argmind-rules/tests/licences.rs` against `docs/licences.toml`.

## What was checked

| | licence | Korean | verdict |
|---|---|---|---|
| Windows 11 natural voices | part of Windows | yes | **free and local; not installed here, so unverified** |
| `qwentts.cpp` + Qwen3-TTS GGUF | MIT engine, Apache-2.0 weights | yes | **builds and speaks — and is four times too slow on CPU** |
| `hexgrad/Kokoro-82M` | Apache-2.0 | **no** | out |
| `myshell-ai/MeloTTS-Korean` | MIT | yes | Python and PyTorch; passed over |
| Piper `ko_KR-kss-medium` | engine MIT, **voice CC BY-NC-SA 4.0** | yes | refused |
| `facebook/mms-tts-kor` | **CC-BY-NC-4.0** | yes | refused |

**The first answer generalised from two data points and was wrong.** Having found Piper's and
MMS's Korean voices both non-commercial, it concluded *the engines are permissive and the voices
are not*. Searching what actually exists turned up Apache-2.0 weights covering Korean in GGUF
with an MIT C++17 runtime. The narrower claim survives: for those two the binding constraint is
the voice licence, and it lives in the model card rather than in the repository.

## qwentts.cpp: it builds here, with one flag

C++17 over GGML — the same GGML `whisper.cpp` already vendors and this project already builds
from source at a pinned commit. Ten languages including Korean, 24 kHz mono, and a `tts-server`
with an OpenAI-compatible HTTP interface, which is the shape argmind already speaks to for
Ollama and for the recogniser.

**MinGW needs `-D_WIN32_WINNT=0x0A00`.** Without it the vendored `cpp-httplib` fails on
`'::CreateFile2' has not been declared`, because MinGW gates that behind a Windows 8 target and
the default here is older. Configure and build otherwise as whisper.cpp does. Checked before
promising anything, which is the discipline that paid for itself on whisper.cpp.

## And then it is four times too slow

`qwen-talker-0.6b-base-Q4_K_M` + `qwen-tokenizer-12hz-Q4_K_M`, 884 MB together, CPU only:

| text | audio | total | RTF |
|---|---|---|---|
| 40 characters | 2.32 s | 9.1 s | **3.92** |
| 119 characters | 8.00 s | 30.7 s | **3.83** |

**MEASURED.** Stable across lengths, so it is a per-frame cost — about 280 ms per frame of
autoregressive decode — rather than a fixed overhead that a longer sentence would amortise.

For comparison, the thing it would replace: **SAPI speaks in 4–43 ms**, and a whole model call
in this agent has a median of 0.61 s. Thirty seconds of synthesis for an eight-second reply is
not a slower conversation, it is a different program.

**Streaming does not save it**, and that is arithmetic rather than opinion. Time to first audio
is a promising 330 ms and the pipeline decodes frame by frame — but at RTF 3.8 the generator
falls behind the speaker by roughly three seconds for every second spoken. Streaming helps when
the generator is faster than real time and nothing else.

## The GPU is there and cannot be used yet

This machine has an **RTX 4090 Laptop GPU**. `nvcuda.dll` and `vulkan-1.dll` are present — the
drivers — but **`nvcc` and `glslc` are not**, so neither GPU backend can be *built*. A CUDA build
needs the CUDA toolkit; a Vulkan build needs the Vulkan SDK for its shader compiler.

`docs/licences.toml` already decided which of those, and decided it before this question came
up: redistributing NVIDIA's runtime libraries is governed by NVIDIA's licence rather than an
open-source one, so **the Vulkan path is the one that redistributes nothing proprietary.**

## What is not known

**Whether the GPU makes it fast enough is unmeasured**, and it is the only question left. A
0.6B model on a 4090 should be well under real time, but *should* is not a number and this
document already contains one thing that reasoning got wrong. The measurement costs an SDK
install and one command.

**Whether a Windows 11 natural voice can be used by an ordinary application here is also
unmeasured.** None is installed — `Speech_OneCore\Isolated` is empty and the only Korean token
is Heami — and the natural voices were Narrator-only when they shipped before being opened up.
It is the cheapest thing to try and it costs two clicks: 설정 → 접근성 → 내레이터 → 자연스러운
음성 추가. `speaker.rs` prefers any Korean voice whose name lacks `Desktop`, so a newer one is
picked up with no code change.

## If the GPU path is taken, the work after it

Not large, and every piece has a precedent in this tree:

1. `Fix::BuildSpeaker` — clone at a pinned commit, configure with the flag above, build, fetch
   two GGUFs with hash checks, run `tts-server` supervised. This is `Fix::BuildRecogniser` with
   different constants.
2. **Playback, which is genuinely new.** SAPI synthesises *and* plays; a neural server hands
   back 24 kHz WAV and something has to put it on the speakers. WASAPI render, mirroring
   `microphone.rs`'s capture — and it has to be interruptible mid-word, because barge-in is what
   `Speaker::stop` means.
