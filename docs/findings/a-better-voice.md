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

## The Vulkan SDK needs administrator rights, and that changes the eventual fix

Downloaded and checked: **274 MB**, `sdk.lunarg.com` over TLS,
`81f474711e9042f4cd22b31b2f7a8870db2e428b21586fb43dd80150be97310d`. It is a Qt Installer
Framework package and it does install headlessly —
`--accept-licenses --default-answer --confirm-command install` — **but every component, including
`com.lunarg.vulkan.core` on its own into a user folder, stops with**

> Cannot elevate access rights while running from command line. Please restart the application as
> administrator.

So this is not `Fix::Install` when it comes to be offered; it is **`Fix::RunElevated`**, where
Windows asks a second time and the person answers twice. That is a different grade in
`fix.rs` and a different sentence in the window, which is exactly why those are separate
variants.

**And the command-line mode cannot raise that prompt itself.** Run unelevated it undoes its own
`Mkdir` and aborts, having printed *restart the application as administrator* — advice it is
not able to take. The wizard, started with no arguments, does raise the prompt. So a fix that
shells out to this installer with flags will fail on a machine where the person has not already
elevated, and the sentence in the window has to say which of the two they are about to get.

A no-administrator route was looked at and put down: `glslc` alone from Google's `shaderc` has
no GitHub release assets, only build-bot URLs that change, and building a fix on those is
building one that breaks quietly. The headers are fine — Khronos, permissive — but the shader
compiler is the piece that is hard to get honestly without the SDK.

## The Vulkan build needs a short path, and that is a constraint on where argmind builds

With the SDK installed, cmake finds Vulkan and reports the good news — `GL_KHR_cooperative_matrix
supported by glslc`, which is the 4090's tensor cores being reachable. Then the build fails, and
not for a reason that looks like a path problem:

> The C compiler ... is not able to compile a simple test program.
> `mingw32-make: CMakeFiles/cmTC_019ae.dir/build.make: No such file or directory`

A compiler that works everywhere else in this tree suddenly cannot build hello world. **It is
Windows' 260-character limit.** `ggml-vulkan` builds its shader generator as a nested external
project, and the try-compile scratch directory inside it lands at

    <wherever you are>/build/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/
    vulkan-shaders-gen-build/CMakeFiles/CMakeScratch/TryCompile-athnoi/...

which is about 130 characters of nesting before the project's own path is counted. Anything
deeper than roughly 120 characters fails, and it fails looking like a broken toolchain.

**This bears directly on `Fix::BuildSpeaker`.** `recogniser::shed()` is
`%LOCALAPPDATA%rgmind\whisper.cpp`, 49 characters, which is fine for whisper.cpp and not
obviously fine here. Whatever builds this has to either sit somewhere short or be measured at
the path it will actually use — and the failure mode is a message about the C compiler, so
nobody would guess.

## What is not known

**Whether the GPU makes it fast enough is unmeasured**, and it is the only question left. A
0.6B model on a 4090 should be well under real time, but *should* is not a number and this
document already contains one thing that reasoning got wrong. The measurement costs one
elevated install and two commands, and it is written out at the end of this file.

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

**Neither is worth writing until the GPU number exists.** A fix that builds a synthesiser which
takes thirty seconds to say a sentence is a button that works and should not be pressed.

## Taking the measurement

The installer is already fetched and verified at the hash above. It needs administrator rights,
so it is two steps rather than one:

```text
# 1. as administrator, once
vulkan_sdk.exe --root C:\VulkanSDK\1.4.357.0 --accept-licenses --default-answer \
               --confirm-command install com.lunarg.vulkan.core

# 2. rebuild with the Vulkan backend and say one sentence
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
      -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00
cmake --build build --config Release -j 8
build/qwen-tts.exe --model qwen-talker-0.6b-base-Q4_K_M.gguf \
                   --codec qwen-tokenizer-12hz-Q4_K_M.gguf --lang Korean -o ko.wav < ko.txt
```

The number to read is `RTF` on the `[Perf] Total` line. **Under 1.0 and this is worth building;
over it and the honest answer is that a neural voice does not fit on this machine yet.**
