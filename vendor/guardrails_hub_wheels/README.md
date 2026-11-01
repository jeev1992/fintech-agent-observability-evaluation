# Vendored Guardrails Hub validators

`guardrails hub install hub://...` fetches a package manifest from
`hub.api.guardrailsai.com` before installing anything. As of 2026-09 that
host no longer resolves (DNS NXDOMAIN) — the command fails for everyone,
regardless of a Guardrails Hub token, because the backend has been
decommissioned. Guardrails Hub's own site now says accounts/tokens aren't
needed at all, but its newer install path (`guardrails_ai.<validator>`
packages) is a different, incompatible layout from the `guardrails.hub`
import style this project uses with the pinned `guardrails-ai==0.6.0`
(pinned in `requirements.txt` to dodge a PyPI quarantine — see git history).

These three `.whl` files are exact repacks of the last validator builds that
were successfully installed via `guardrails hub install` before the backend
died. They are plain, offline-installable wheels with no further dependency
on any Guardrails Hub service:

- `guardrails_grhub_regex_match-0.0.0-py3-none-any.whl`
- `guardrails_grhub_toxic_language-0.0.2-py3-none-any.whl`
- `guardrails_grhub_competitor_check-0.0.2-py3-none-any.whl`

Install them with:

```bash
pip install --no-deps vendor/guardrails_hub_wheels/*.whl
```

`--no-deps` is important — these wheels declare `guardrails-ai` as a
dependency without a version pin, which would otherwise pull in a newer,
incompatible `guardrails-ai` release over the pinned `v0.6.0`.
