"""What the shipped Python calls a turn, and what it throws away.

**The guard is a real instrument and it was nearly lost.** `argmind/env/whisper.py` catches
Whisper's degenerate loops — 7 of 7 reproduced — and rejects none of the strings that must survive.
Nothing on the Rust side did that until it was ported, and a port of a guard is the kind of thing
that looks right and is off by one character somewhere: Python slices by character, and a byte
window over `한글자막 by ` measures a different thing entirely.

So every corpus string is written down here with the Python's own verdict, and the Rust replays it.
The interesting cases are not the obvious ones — they are `Thank you. Thank you.` (two copies, a
person saying 네, 네) surviving, and `아니 그거 말고` three times surviving because it is under the
floor while the same phrase five times over is not.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from argmind.env.whisper import (                                          # noqa: E402
    REPEAT_FLOOR, REPEAT_PROBE, REPEAT_TIMES, _repeating, _wordless, why_rejected)

#: Verbatim from the runs recorded in `argmind/env/whisper.py`. The two long ones are the same
#: strings rebuilt from their cycle: 63 copies cut mid-phrase, and 20 of the 40 Whisper emitted.
DEGENERATE = [
    "Thank you. Thank you.",
    "... ... ... ... ... ... ...",
    "... ... ... ... ... ... ... ... ...",
    "うんご視聴ありがとうございましたご視聴ありがとうございましたご視聴ありがとうございました",
    "Tchau, tchau. Tchau, tchau. Tchau, tchau. Tchau, tchau.",
    "Tchau, tchau. Se inscreva no canal e ative o sininho para receber notificações de novos vídeos.",
    "곱창 눌러줘 " + "한글자막 by " * 20,
    "Thank you. " + "I'm going to make sure that " * 63 + "I'm going",
    "Mm-hmm Thank you. Thank you. Thank you.",
    "속삭이냐 좋아 다음 영상에서 만나요. 한글자막 by 한효정",
]

#: Real transcripts of real synthesised speech, from the round trip in `whisper.py`'s docstring.
CLEAN = [
    "지우개 눌러줘.", "빨간색을 선택할 수 있어요.", "네, 이미지를 저장하고 창을 닫을게요.",
    "화면에 20개의 버튼이 있어요",
    "공번창에서 특수문자를 선택한 후 3번 버튼을 클릭해 선택하고 4번 버튼으로 복사하세요.",
    "Okay, I'll select the red color.", "copy is number four",
    "I don't have a history of actions, you'll need to describe what you are trying to accomplish.",
    "BIOS version slash date is October 22nd, 2025.",
]

#: A person repeating themselves, which is not a degeneration. These are the ones that cost a turn
#: if the guard is wrong, and a rejected turn is silent while a fabricated one at least shows up.
REPETITIVE_HUMAN = [
    "네, 네, 네.", "아니 그거 말고", "아니 그거 말고 아니 그거 말고 아니 그거 말고",
    "그거 그거 그거요", "다시, 다시.", "hello hello hello",
    "no no no no", "okay okay", "잠깐만요 잠깐만요",
]

#: The edge the floor exists for, and the one just past it.
EDGES = [
    "아니 그거 말고 아니 그거 말고 아니 그거 말고",
    "아니 그거 말고 아니 그거 말고 아니 그거 말고 아니 그거 말고 아니 그거 말고",
    "곱창 눌러줘 한글자막 by 한글자막 by 한글자막 by 한글자막 by 한글자막 by",
    "",
    "   ",
    ".",
    "...",
    "42",
    "3.14!",
]


def rows(texts):
    return [{"text": t,
             "wordless": _wordless(t),
             "repeating": _repeating(t),
             #: The shipped default: no pace limit, so pace rejects nothing.
             "why": why_rejected(t),
             #: And with one set, to prove the limit is wired rather than merely present. One
             #: second of audio, so the pace is the character count.
             "why_at_5_per_second": why_rejected(t, 1.0, 5.0)}
            for t in texts]


out = {
    "constants": {"floor": REPEAT_FLOOR, "probe": REPEAT_PROBE, "times": REPEAT_TIMES},
    "degenerate": rows(DEGENERATE),
    "clean": rows(CLEAN),
    "repetitive_human": rows(REPETITIVE_HUMAN),
    "edges": rows(EDGES),
}

at = pathlib.Path(__file__).with_suffix(".json")
at.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
caught = sum(1 for r in out["degenerate"] if r["why"])
print(f"   hearing.json — {caught} of {len(DEGENERATE)} degenerate rejected, "
      f"{sum(1 for k in ('clean', 'repetitive_human') for r in out[k] if r['why'])} "
      f"of {len(CLEAN) + len(REPETITIVE_HUMAN)} survivors rejected")
