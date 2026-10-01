# Browser Automation Agent

An AI agent that operates a real web browser the way a person does. You describe
a task in plain language; it plans an approach, drives Chromium to carry it out,
pauses to ask before doing anything irreversible, and reports what it found.

```
> Find the cheapest RTX 5070 laptop under ₹1.2L
> Compare MacBook Air prices on Amazon and Flipkart
> What are the top 3 stories on Hacker News right now?
> Log into GitHub and create a new repository
```

It is built around one idea: **the model decides what to do, and Python decides
what is allowed to happen.** Every action passes through a safety gate before it
touches the browser, and anything that spends money, deletes something, or sends
a message stops and waits for you.

📐 **[Read the architecture document](docs/ARCHITECTURE.md)** for how it works
and why it is built this way.

---

## How it sees the page

The agent does not guess CSS selectors and does not read raw HTML. Every
interactive element on the page is found, checked for visibility *and*
clickability, and given a number:

```
Gaming laptops
[0]<input placeholder="Search" type="text"></input>
[1]<button>Search</button>
ASUS ROG Strix G16 — RTX 5070
₹1,14,990
[2]<a href="/dp/B0F...">See product details</a>
[3]<button aria-label="Add ASUS ROG Strix to cart">Add to cart</button>
```

The model replies `click(index=3)`. When the text alone is not enough — a
canvas-drawn app, an icon-only button — it gets a screenshot with **the same
numbers drawn on it**, so both views describe the same world in the same terms.

---

## Quick start

### Requirements

- **Python 3.12+** (developed on 3.14)
- **Node.js 20+** for the web UI
- **Docker** for Postgres — optional, see below

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
playwright install chromium
```

### 2. Configuration

```bash
cp .env.example .env            # from the repository root
```

The defaults work as-is. `LLM_PROVIDER=mock` runs a keyless, offline agent that
exercises the entire system — useful for confirming your setup before you have
an API key. For real reasoning, get a free key from
[Google AI Studio](https://aistudio.google.com/apikey) and set:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```

For adaptive no-cost routing, set `LLM_PROVIDER=adaptive`, add any Gemini,
Groq, OpenRouter, and NVIDIA API keys you have to `.env`, and keep Ollama
running locally. Routine browser tasks try Ollama first; complex tasks
prioritize Gemini. NVIDIA Muse Glimmer, GLM-5.3-Flash, and Nemotron are added
as capability-filtered fallbacks. A rate-limit, network, or server failure
immediately moves the current request to the next eligible provider. Gemini
accepts up to three keys and rotates them per request. Provider settings are in
`.env.example`.

### 3. Database (optional but recommended)

```bash
docker compose up -d
```

Brings up Postgres with pgvector for task history, the audit trail, and
long-term memory. **Skip it and everything still works** — history simply lives
in memory until the server restarts, and long-term memory is disabled.

### 4. Run it

```bash
# Terminal 1 — backend
cd backend && python -m app.main

# Terminal 2 — UI
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**. A Chromium window opens when the first task
starts — that is the agent working, and you can watch it.

---

## Using it

### Web UI

Two panes: the agent's reasoning on the left, what it is looking at on the
right. Approval prompts appear over the screenshot they refer to, so you are
never asked to approve an action you cannot see.

### Command line

Same graph, same tools, no web stack:

```bash
cd backend
python -m app.cli "find the cheapest 27 inch 4k monitor"
python -m app.cli "check my github notifications" --profile work
python -m app.cli "what is the weather in Pune" --quiet
```

Approvals are answered at the prompt. Anything other than an explicit `y` is
treated as no.

### HTTP API

```bash
# Start a task
curl -X POST localhost:8000/api/tasks \
     -H 'Content-Type: application/json' \
     -d '{"goal": "find the top story on Hacker News"}'

# Watch it
websocat ws://localhost:8000/ws/tasks/{task_id}

# Answer an approval
curl -X POST localhost:8000/api/tasks/{task_id}/approval \
     -H 'Content-Type: application/json' -d '{"approved": true}'
```

Interactive docs at **http://localhost:8000/docs**.

| Endpoint | Purpose |
|---|---|
| `POST /api/tasks` | Start a task; returns immediately with an id |
| `GET /api/tasks/{id}` | Status, plan, result |
| `GET /api/tasks/{id}/events` | Full narration, replayable |
| `GET /api/tasks/{id}/tool-calls` | Audit trail with risk ratings |
| `POST /api/tasks/{id}/approval` | Approve or reject a pending action |
| `POST /api/tasks/{id}/cancel` | Stop a run and close its browser |
| `GET /api/health` | What is actually wired up right now |
| `WS /ws/tasks/{id}` | Live event stream |

---

## Staying logged in

Sessions use a **persistent Chromium profile**, so cookies and logins survive
between runs. Log in once, by hand, and the agent inherits it:

```bash
python -m app.cli "open github.com" --profile work
# log in yourself in the window that opens, then Ctrl-C

python -m app.cli "create a repo called scratch" --profile work   # already signed in
```

**The agent never types passwords, card numbers, CVVs or one-time codes.** Those
fields are blocked outright — not "ask first", blocked. If a task needs
credentials, it stops and asks you to enter them yourself.

Profiles live in `data/profiles/`, which is gitignored. Treat it as sensitive.

---

## Safety

Every proposed action is classified before it runs.

| Level | Examples | Behaviour |
|---|---|---|
| **Safe** | navigate, scroll, read, search | Runs |
| **Low** | click a link, fill a search box | Runs |
| **Confirm** | *Place your order*, *Delete repository*, *Send message* | **Pauses and asks you** |
| **Blocked** | password fields, `file://`, `chrome://` | Refused, and the model is told why |

Classification reads the *label of the element* — "Place your order" is the
signal, not the fact that a click happened — and it errs toward asking. On a
checkout page even a bland "Continue" requires approval.

Rejecting does not end the task. The agent is told no and looks for another way.

Optional domain restrictions:

```ini
ALLOWED_DOMAINS=amazon.in,flipkart.com   # empty = any site
BLOCKED_DOMAINS=facebook.com
```

---

## Configuration

Everything lives in `.env`. The knobs that matter most:

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `adaptive` for Ollama/Gemini/NVIDIA/Groq/OpenRouter routing; `gemini`, `groq`, `openrouter`, `ollama`, or `mock` for one provider |
| `GEMINI_API_KEY` / `_2` / `_3` | — | From [AI Studio](https://aistudio.google.com/apikey); secondary keys rotate per request |
| `GROQ_API_KEY` | — | Groq API key for hosted routing |
| `OPENROUTER_API_KEY` | — | OpenRouter API key; defaults to its free-model router |
| `NVIDIA_API_KEY` | — | NVIDIA Build API key; enables the tested Muse, GLM, and Nemotron models in adaptive routing |
| `BROWSER_HEADLESS` | `false` | `false` lets you watch it work |
| `BROWSER_DEFAULT_PROFILE` | `default` | Which saved profile to use |
| `VISION_MODE` | `auto` | `auto` sends screenshots only when text perception is failing |
| `AGENT_MAX_STEPS` | `40` | Hard ceiling on actions per task |
| `AGENT_MAX_FAILURES` | `8` | Give up after this many failed actions |
| `REQUIRE_APPROVAL` | `true` | Turning this off is not recommended |
| `TAVILY_API_KEY` | — | Optional; without it, search runs through the browser |

`VISION_MODE=auto` is the cost-sensitive default: screenshots cost ~1,200 tokens
each, so they are spent only when the DOM looks unhelpful or the last action
failed. Use `always` for stubborn sites, `never` to minimise cost.

---

## Testing

```bash
cd backend
pytest                              # everything
pytest --ignore=tests/test_dom_index.py   # fast: skips the real-browser tests
```

**96 tests.** They fall into three groups:

- **Graph behaviour** — the real graph, router, safety policy and tool
  dispatcher, driven by a scripted model and a fake browser. Verifies that a
  purchase button pauses, that rejection is reported rather than executed, that
  a runaway agent is stopped, and that only an explicit yes counts as approval.
- **DOM indexing** — against real headless Chromium. Verifies that invisible,
  disabled and *covered* elements are never offered to the model, that shadow
  DOM is traversed, and that credential values cannot leak through any of the
  three paths that could carry them.
- **Memory** — against real Postgres + pgvector, skipped automatically when the
  database is not running.

Nothing needs an API key or network access to a model.

---

## Project layout

```
├── backend/
│   ├── app/
│   │   ├── agent/          LangGraph orchestration: nodes, state, routing, prompts
│   │   ├── browser/        Perception (DOM indexer) and action (Playwright verbs)
│   │   ├── tools/          The 21 tools the model can call
│   │   ├── llm/            Provider-agnostic interface + Gemini + mock
│   │   ├── safety/         What needs approval, what is forbidden
│   │   ├── memory/         Long-term facts over pgvector
│   │   ├── db/             Task history and audit trail
│   │   ├── events/         The narration bus
│   │   └── api/            REST + WebSocket
│   └── tests/
├── frontend/               Next.js UI: timeline, live preview, approvals
├── docs/ARCHITECTURE.md    How it works, and why
├── docker-compose.yml      Postgres + pgvector
└── .env.example
```

---

## Troubleshooting

**"mock LLM" in the status bar** — set `LLM_PROVIDER=adaptive` and configure
provider keys in `.env`, or choose a single provider, then restart the backend.

**"no database"** — run `docker compose up -d`. Everything works without it; you
lose history across restarts and long-term memory.

**`Executable doesn't exist`** — run `playwright install chromium`.

**`Fatal error in launcher: Unable to create process using '...python.exe'`** —
the project folder was renamed after the virtualenv was created. Virtualenvs are
not relocatable: every console script (`pip.exe`, `playwright.exe`, `pytest.exe`)
bakes in an absolute path to `python.exe`, and renaming the folder invalidates
all of them at once. `python.exe` itself still works, so the quickest check is:

```bash
.venv/Scripts/python.exe -m pip --version   # works
.venv/Scripts/pip.exe --version             # "Fatal error in launcher"
```

Repair the shims in place, without reinstalling everything:

```bash
.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps pip playwright
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Or delete `.venv` and start over. If `python -m venv .venv` then fails with
`Unable to copy ... venvlauncher.exe`, something is holding `python.exe` open —
usually the VS Code Python extension's language server. Close the editor (or
reload the window) and retry.

As a habit, `python -m pip` and `python -m playwright` are immune to this class
of problem, since they never go through a shim.

**The agent keeps failing on one site** — try `VISION_MODE=always`. Some sites
are genuinely opaque to DOM inspection. Sites with aggressive bot detection may
refuse a fresh profile; logging in by hand once often fixes it.

**Postgres port 5433 is taken** — change the host port in `docker-compose.yml`
and update `DATABASE_URL` to match. It is 5433, not 5432, specifically to avoid
colliding with a local Postgres.

---

## Where it stops

Worth knowing before you rely on it:

- **CAPTCHAs are not solved.** The agent recognises it is stuck and asks you.
- **Prompt injection is mitigated, not eliminated.** A page can address the
  model directly. Because the safety gate runs in Python on the proposed action,
  an injected instruction cannot talk its way past the approval prompt — but it
  can still waste steps or steer the agent somewhere unhelpful.
- **The approval classifier is a heuristic**, not a proof. It catches the
  buttons that matter and produces some false positives.
- **No API authentication.** It binds to localhost and assumes a single-user
  machine.

See [Limitations](docs/ARCHITECTURE.md#9-limitations) for the full list.
