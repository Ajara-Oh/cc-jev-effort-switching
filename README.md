# claude-jev

Runs Claude Code through a local proxy that asks [Jev](https://typesafe.ai) which
reasoning effort each step needs, and switches effort mid-conversation without
breaking prompt caching.

## How it works

Before each model call, the proxy sends Jev a short summary of the task: the
latest prompt, the recent tool calls and results, and how many results in a row
looked like errors. Jev picks one of the configured effort levels. If the pick
differs from the current level and Jev is confident enough, the proxy inserts an
effort-only system message (beta `mid-conversation-output-config-2026-07-01`)
before the newest message.

Each marker is stored under a hash of the messages before it and re-inserted at
the same place on every later request, so the already-cached prefix never
changes. Markers are kept in memory; after a proxy restart, a running session
pays one cache write.

On any failure (no key, Jev timeout or error, unsupported model, or the API
rejecting the modified request) the original request is sent unchanged.

Supported models: Opus 5 / 5.5, Fable 5.1 / 5.2, Mythos 5.1.

## Setup

Python 3, standard library only.

```sh
git clone <this repo> ~/claude-jev
cp ~/claude-jev/config.example.json ~/claude-jev/config.json   # add your Jev key
ln -s ~/claude-jev/claude-jev ~/.local/bin/claude-jev
claude-jev            # any arguments are passed to claude
```

The first launch starts a detached supervisor that keeps the proxy running on
port 37491 (`JEV_PORT` to change it), so background sessions keep working after
the terminal that started them closes.

## Config

`config.json` next to the scripts (or the path in `JEV_CONFIG`):

| key | default | |
|---|---|---|
| `api_key` | `$JEV_API_KEY` | Jev key; without one the proxy only passes requests through |
| `levels` | `low, medium, high, xhigh` | levels Jev may choose from (`max` is also available) |
| `min_confidence` | `0.45` | below this, Jev's pick is ignored |
| `jev_timeout_s` | `3.0` | time allowed for each Jev call |
| `debug` | `false` | log every decision to `effort.log` (`JEV_DEBUG=1` also works) |
| `upstream` | `https://api.anthropic.com` | or `$JEV_UPSTREAM` |

## Tests

```sh
python3 -m unittest test_proxy
```
