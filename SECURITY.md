# Security

argmind presses buttons, types and runs commands on the computer it is installed on. A flaw that
lets something other than the person at that computer decide what it does is the most serious kind
of bug it can have.

## Reporting

**Please report privately**, through GitHub's *Report a vulnerability* button on this repository's
**Security** tab. Do not open a public issue for it.

This is one person's side project. There is no bounty and no promised response time. A report is
read, answered when it can be, and fixed in the open once a fix exists.

## What counts

Anything that lets a party other than the person at the keyboard make argmind act, read, or send
something they did not allow. For example:

- **The local server.** It listens on loopback only, wants a token minted each run, and refuses a
  request whose `Host` is not this machine (the DNS-rebinding guard in `argmind-serve`). A way past
  any of these counts.
- **A page in argmind's own browser pane, a document a task reads, or a window title** getting
  argmind to press, type or run something the person did not ask for (prompt injection that crosses
  a gate).
- **The task sandbox.** The `argmind-sandbox` WSL distro has drive mounting and Windows interop off
  and mounts only the work folder. A way to reach the rest of the disk or a Windows program from
  inside it counts.
- **The work desktop.** Commands a task runs start on a desktop and window station of argmind's
  own. Programs it opens on its work screen run on a desktop of its own, in a job object, with the
  clipboard held shut while it presses. Reaching the person's desktop, clipboard or input from
  either counts.
- **The person's own limits.** Typing into a password field, acting against a *하지 않음* (never)
  rule, or acting without confirmation when *확인 없이 실행* is off.
- **Anything leaving the machine** that the README's *What leaves the machine* section does not list.

## What does not

- **The model choosing the wrong control.** That is a measured limit (21 of 28 answerable goals,
  qwen3:8b), not a vulnerability, unless it gets past one of the gates above.
- **Something the person turned on doing what the switch says.** For example, *스스로 실행* (act on
  argmind's own ideas) acting on argmind's own idea.
- **Tasks' PowerShell commands.** They run as the person, on a separate desktop and window station,
  and the README says this is weaker than the WSL sandbox. A way past the separate desktop or window
  station is in scope; a command reaching the person's files is not, because it is documented.

## Supported versions

Only the latest commit on `main` and the latest release.
