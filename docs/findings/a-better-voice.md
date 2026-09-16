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

**The number here was an estimate and it has since been measured. It was too kind.** Counting
the tree a successful build leaves behind and adding back the try-compile path from the failing
one: `ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/CMakeFiles/`
is 87 characters below the build folder, and
`CMakeScratch/TryCompile-xxxxxx/CMakeFiles/cmTC_xxxxx.dir/build.make` adds 68 more — **155, not
130.** The run that failed was 126 + 155 = 281; the one that works is 52 + 155 = 207. So the
ceiling on the build directory is about **104 characters**, not 120. A correction in the
direction that makes the constraint tighter, which is the direction worth checking twice.

**This bears directly on `Fix::BuildSpeaker`.** `recogniser::shed()` is
`%LOCALAPPDATA%/argmind/whisper.cpp`, 49 characters, which is fine for whisper.cpp and not
obviously fine here. Whatever builds this has to either sit somewhere short or be measured at
the path it will actually use — and the failure mode is a message about the C compiler, so
nobody would guess. It was measured before the automation was written rather than after:
`speaker::shed()` plus its build folder `b` is **52**, and a full build was run there by hand
first.

## The GPU is thirty-two times faster and the output is wrong

With the SDK installed and both traps cleared, the Vulkan build runs on the right device —
`NVIDIA GeForce RTX 4090 Laptop GPU ... matrix cores: NV_coopmat2`, the tensor cores.

| backend | RTF | frames | audio from a 40-character sentence |
|---|---|---|---|
| CPU | 3.92 | 25, **stops at EOS** | 2.32 s |
| CPU, same Vulkan-built binary | 4.81 | 33, **stops at EOS** | 2.64 s |
| **Vulkan** | **0.123** | **2048, the `--max-new` cap** | **163.84 s** |

Eight times faster than real time, from four times slower — a **thirty-two fold** speedup, and
the number is real throughput. **The synthesis is not.** A forty-character sentence became two
and a half minutes of audio, because the talker never emitted an end-of-speech token and ran
until the cap stopped it.

**That flattering number arrived attached to a broken run, which is the shape this project's
standards exist for.** The RTF would have gone straight into a table if the frame count had not
been sitting next to it.

### It is the backend, isolated

**The same binary**, with `GGML_VK_VISIBLE_DEVICES=""` so it falls back to CPU, stops at EOS at
step 33. So it is not the build, not the flags, not the model file and not the text: it is
`ggml-vulkan` computing something different enough that the EOS logit never wins.

That is a useful shape of bug to know about, because it is silent. Nothing errors, nothing
warns, and the only visible symptom is that the audio is far too long — which a caller with a
timeout would report as *slow*.

### What is being tried next

`Q4_K_M` is a K-quant, and K-quant kernels are the most involved path in any backend. `Q8_0` is
a far simpler one. If Q8_0 on Vulkan stops at EOS then the fault is in the K-quant path and the
answer is *use Q8_0*, at 993 MB instead of 629 MB. If it does not, the fault is deeper and this
wants reporting upstream rather than working around.

## What is not known

**Whether a Windows 11 natural voice can be used by an ordinary application here is also
unmeasured.** None is installed — `Speech_OneCore\Isolated` is empty and the only Korean token
is Heami — and the natural voices were Narrator-only when they shipped before being opened up.
It is the cheapest thing to try and it costs two clicks: 설정 → 접근성 → 내레이터 → 자연스러운
음성 추가. `speaker.rs` prefers any Korean voice whose name lacks `Desktop`, so a newer one is
picked up with no code change.

## The GPU path is taken. The work after it

Not large, and every piece has a precedent in this tree:

1. `Fix::BuildSpeaker` — clone at a pinned commit, configure with the flags above, build, fetch
   two GGUFs with hash checks, run `tts-server` supervised. This is `Fix::BuildRecogniser` with
   different constants, **plus a short build path and the toolchain bin first**.

### The client exists and speaks — 2026-09-16

`argmind-env-qwentts`, the same shape as the Ollama and whisper clients: raw HTTP/1.1, no crate.
`POST /v1/audio/speech` with `response_format: "wav"`, and `argmind_voice::Speaker` on top of the
loudspeaker.

Measured end to end through the server, warm:

| | |
|---|---|
| `say()` returns | **0 ms** — the request is on its own thread, because a blocking one cannot be interrupted |
| `say()` to the first sound | **~1.8 s** for a reply-length sentence |
| synthesis itself | 1.80–1.81 s for 8.2–9.7 s of audio, RTF **0.19–0.22** |
| interrupting mid-sentence | works; silent within 0.6 s of `stop()` |

**The first utterance after the server starts costs far more**: 9.7 s, of which `CodecDecode`
alone was 7.77 s against 0.32 s on the next request. Vulkan compiling its pipelines, paid once
by a server that stays up — and a reason the server is the right shape rather than the
command-line tool, which would pay it every time.

**`say()` to first sound is the number that matters and 1.8 s is not good.** SAPI starts in
4–43 ms. The model call before it has a median of 0.61 s, so the agent would sit silent for
about two and a half seconds after a question. That is noticeable and it is not what this
project means by real-time interaction.

### Streaming, and now it is a number

The hypothesis was that reading the body incrementally would put first sound in the hundreds of
milliseconds instead of 1.8 s. Built and measured, four runs:

| | |
|---|---|
| `say()` to **first sound** | **27, 34, 233, 238 ms** |
| synthesis, whole utterance | 1.30–1.55 s for 8.4–8.7 s of audio |
| RTF | 0.15–0.18 |

**Eight to sixty times better, and it changes the answer.** With the model call's median of
0.61 s in front of it, the agent starts speaking about 0.65–0.85 s after a question — against
SAPI's 0.65 s. The neural voice now costs a couple of hundred milliseconds over the old one at
worst, and nothing at all once the server is warm.

It also stays ahead: 1.3 s of synthesis for 8.6 s of audio means the generator is six times
faster than the speaker consumes, so there is no chance of running dry mid-sentence.

**Three things the streaming path needed that the whole-file one did not.**

`Transfer-Encoding: chunked`, because a server that decodes as it goes cannot know the length.
A chunk decoder, twenty lines, and the `0`-length chunk that ends it.

**A sample can straddle a chunk boundary.** The server sends bytes and does not care where a
sixteen-bit sample begins, so an odd byte at the end of one piece is the first half of a sample
completed by the next. Dropping it would be a click in every chunk.

And the loudspeaker had to learn the difference between *out of audio* and *out of utterance*.
An empty queue mid-stream is the synthesiser still working; resetting the device on it would put
a click between every pair of pieces. `finished()` is what says the utterance is over, and until
it is called the feeder waits.
2. **Playback, which is genuinely new.** SAPI synthesises *and* plays; a neural server hands
   back 24 kHz WAV and something has to put it on the speakers. WASAPI render, mirroring
   `microphone.rs`'s capture — and it has to be interruptible mid-word, because barge-in is what
   `Speaker::stop` means.

**The number exists now and it says go ahead** — Q8_0, Vulkan, through `tts-server` rather than
the command line.

### Playback, and what barge-in costs

Written: `argmind-env-windows/src/loudspeaker.rs`, the mirror of the microphone. Measured with
`--example bargein`, twice:

| | |
|---|---|
| `play()` to sound going out | **3–4 ms** |
| `stop()` to the device reset | **4–6 ms** |
| `play()` over something already speaking | **1 ms** to return |

**Bounded rather than exact.** Those are times to the feeder thread having stopped and reset the
device; what WASAPI does in the microsecond after `Reset` is not measured, and the device buffer
is 60 ms, so *heard* silence is somewhere under that. Measuring it properly wants loopback
capture — recording what the speakers actually output — which is the same shape as
`microphone.rs` and has not been built.

**Two things in that file were wrong before they were measured.** The module comment said `stop`
tells the device to drop what it is holding; it did not, it only cleared the queue, which is a
stop that is late by a buffer and that nobody would notice. And `stop` set the `speaking` flag
itself, so *how long does stopping take* was timing one line setting a boolean — it read 0 ms
every time, whatever the device was doing. Both are fixed and both are written down where they
happened.

A third turned up in the suite: the output frame count was `(len / ratio).floor()`, which loses
a frame at 44.1 kHz because 24000 / (24000/44100) comes out as 44099.999…. Integer arithmetic
now.

## argmind builds it itself, and three things had to be found first — 2026-09-16

`Fix::BuildSpeaker` is the recogniser's fix with different constants, and writing it turned up
three failures that the recogniser's build could not have found because it does not do the
things that cause them.

**A process cannot see an environment variable set after it started.** This is not obscure and it
is easy to miss, because the sequence that hits it is the ordinary one: argmind says the Vulkan
SDK is missing, the person installs it, and argmind — the same process, four minutes later —
still has the environment it was born with. `glslc --version` answers *not installed* on a
machine where it is. So the SDK is found through `reg query` against
`HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment`, which is the key the
installer wrote to, and the directory is then checked on disk because an uninstall leaves the
variable behind. **Measured on this machine**: `VULKAN_SDK` is empty in a shell started before
the install and `C:\VulkanSDK\1.4.357.0` in the registry.

**The value must be taken to the end of the line, not to the next space.** `reg.exe` prints
`    NAME    REG_SZ    C:\Program Files\...`, and splitting on whitespace turns a default
install into `C:\Program`. Nothing about that fails where it happened; the build fails later,
saying something about the compiler. It is its own function, `building::value_in`, so it can be
given text rather than a machine.

**The compiler's own directory belongs at the front of the child's PATH, always.** This is the
`0xC0000139` failure from further up, and the fix is the same fix `bring_runtime` already applies
to the shipped binary — say which directory rather than hope. It is now in `Recipe::dress` and
therefore applies to the recogniser's build too, which did not need it and is not harmed by it.

**What is offered, and what is separate.** The SDK and the voice are two needs with two keys,
not one. They look like one decision and they are not: the SDK is a machine-wide developer kit
installed with administrator rights, and folding it into the voice would mean a remembered
*항상허용* for "the voice" silently covering an elevated install. `winget install --id
KhronosGroup.VulkanSDK -e --interactive` is the command — **`--interactive` because the one
headless attempt on this machine ended in `Installation aborted!`**, so letting the installer put
its own window up is the path with evidence behind it. The package id and its `MIT OR Apache-2.0`
licence were read from `winget show`; **the install itself is unverified here**, because the SDK
was already on this machine and uninstalling it to test the button was not worth the risk.

**`MINUTES = 15`, and the measured number is 7.** A fresh clone with submodules and a full build
at `-j 8`, at `%LOCALAPPDATA%rgmind\qwentts.cpp`, which is the path it will actually use. The
figure is for every target rather than the one the fix asks for. Fifteen is what a person is told
before they commit an evening and is deliberately not the measured number, for the same reason
`recogniser::MINUTES` is 10 against a measured 4.

**The talker is Q8_0 and the codec stays Q4_K_M.** That pairing is the one measured end to end,
and it is not tidied up to match. The K-quant defect was in the talker's sampling loop; the codec
is a different graph and every run produced correct speech, which is an observation and not a
proof. A Q8_0 codec is published and has not been tried. `tests/speaker.rs` asserts the pairing,
because an edit that made them consistent would reintroduce a voice that never stops talking.

## Taking the measurement

The installer is already fetched and verified at the hash above. It needs administrator rights,
so it is two steps rather than one:

```text
# 1. as administrator, once — the wizard, not the command line: the CLI mode cannot elevate
vulkan_sdk.exe

# 2. somewhere SHORT. C:/q, not a hundred-character scratch directory
git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git C:/q/t

# 3. the toolchain's own bin FIRST on the path, or the shader generator loads the wrong
#    libstdc++ and dies with 0xC0000139
export PATH="<winlibs>/mingw64/bin:C:/VulkanSDK/1.4.357.0/Bin:$PATH"
cmake -B b -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
      -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00
cmake --build b --config Release -j 8

# 4. say one sentence
b/qwen-tts.exe --model qwen-talker-0.6b-base-*.gguf \
               --codec qwen-tokenizer-12hz-*.gguf --lang Korean -o ko.wav < ko.txt
```

**Read two numbers, not one.** `RTF` on the `[Perf] Total` line says whether it is fast enough;
the frame count beside it says whether it is right. A run that hits `--max-new` never found the
end of the sentence, and its RTF is the throughput of generating nonsense.
