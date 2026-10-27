# HCP server

```bash
uv run harbor serve --data-dir .harbor-serve
```

API: `http://127.0.0.1:8000/hcp/v0/<method>`.
Interactive documentation: `http://127.0.0.1:8000/docs`.

Submit a `JobConfig` or `JobLock` to run existing Harbor agents and sandboxes.
Tasks must be accessible to the server; install provider extras as needed.

- `--data-dir` holds `runtime.sqlite3` and output files under `files/`.
  Artifact URIs are returned; files are not served by HCP.
- Set `HARBOR_API_KEY` for bearer authentication; required when binding beyond localhost.
- `--max-concurrent` limits running rollouts across jobs (default: 128).
- Run one server process per data directory.

```bash
uv run pytest tests/unit/test_hcp_server.py
```

## Inference relay

```bash
uv run harbor serve --host 0.0.0.0 \
  --relay-url https://your-server.example/relay/v0/inference
```

Set `HARBOR_API_KEY` first. The URL must reach this server from the sandbox;
configure HTTPS/public ingress separately when self-hosting.

Set `relay_inference: true` on an agent in the submitted job config (or lock).
Harbor injects a shared base URL and a random token for each rollout. The relay
supports agents with configurable OpenAI/Anthropic endpoints and API keys,
including Claude Code and Codex. It forwards bytes without translating APIs.

The training-side connector authenticates with `Authorization: Bearer <HARBOR_API_KEY>`:

| Submission | WebSocket |
| --- | --- |
| No relay | None |
| Relay enabled | `/relay/v0/connect/jobs/{job_id}` |
| Relay enabled, `session_id` supplied to `submit_job` | `/relay/v0/connect/sessions/{session_id}` |

Generate a fresh session ID per training run to share a connection across jobs.
One active connection is allowed per job/session. Shared sessions stay open when
individual jobs finish; per-job connections close with `job_complete`.

Relay → connector (same format for job and session connections):
```json
{"type":"request","rollout_id":"…","request_id":"…","method":"POST","target":"/v1/messages","headers":[["content-type","application/json"]],"body_b64":"…"}
```

Connector → relay (same `request_id` throughout):
```json
{"type":"response_start","request_id":"…","status":200,"headers":[["content-type","text/event-stream"]]}
{"type":"response_chunk","request_id":"…","body_b64":"…"}
{"type":"response_end","request_id":"…"}
```

The connector forwards to its proxy, keeps upstream credentials there, and records
`rollout_id`/`request_id` as capture metadata. A connector failure uses
`{"type":"error","request_id":"…","code":"…","message":"…"}`; ordinary HTTP errors
use the normal response sequence. Agent credentials and hop-by-hop headers are stripped.

Cancellation/timeouts send `{"type":"cancel","request_id":"…","reason":"…"}`.
Disconnects fail in-flight requests; reconnect for future requests without replaying.
Requests wait up to 30 seconds for a connector. Inference has no total deadline by
default; I/O waits time out after 30 minutes. Payload sizes, concurrency, and response
buffers remain bounded. Tokens are revoked when a rollout ends or is hard-cancelled;
soft cancellation lets running inference finish. Retries receive fresh tokens.
