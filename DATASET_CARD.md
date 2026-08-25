# Dataset Card: Public WildChat Realism Subset

## Summary

`data/wildchat_blog_20.jsonl` is a safety-reviewed subset derived from
[allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M).
The filename is retained for compatibility with the paper-time evaluation, but
the public file contains 19 conversations. Twenty candidate conversations were
reviewed for release; one was excluded because actionable account- or
system-compromise material could not be removed without destroying the
conversation's research meaning.

This subset supports the repository's realism evaluation. It is not intended
to represent WildChat-1M, model users, or deployment traffic generally.

## Source and license

WildChat-1M is distributed by its maintainers under the Open Data Commons
Attribution License ([ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)).
The WildChat-derived records in this repository remain subject to those terms,
not the repository's root MIT license. See `THIRD_PARTY_NOTICES.md`.

Please cite the source paper when using this subset:

> Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and
> Yuntian Deng. "WildChat: 1M ChatGPT Interaction Logs in the Wild."
> ICLR 2024. [Paper](https://openreview.net/forum?id=Bl8u7ZRlbM).

## Historical selection and evaluation processing

The repository records that the paper-time realism haystack contained 20
WildChat conversations. It does not contain enough provenance to establish how
those candidates were sampled from WildChat-1M, so this release does not claim
that the candidates were random, stratified, or representative.

For topic matching, the candidate conversations were scored against 181 Petri
seeds on topic, tone, behavior, structure, and distinguishability. A Hungarian
assignment selected a one-to-one set of seeds. The public release filters the
existing score cache to the 19 retained conversations, reindexes release-local
rows, and recomputes that assignment without making new model-provider calls.

The inherited retained cache contains 3,428 unique conversation/seed pairs out
of the complete 19 × 181 = 3,439 matrix. Eleven pairs were already absent from
the paper-time cache. The realism driver logs this incomplete coverage and
zero-fills each of the 11 missing scores with exactly 0.0 before running the
Hungarian assignment; it does not silently make provider calls to fill them.
This conservative zero-fill policy is part of the public rerun configuration,
and results may differ if users explicitly recompute the missing scores.

## Release processing

The release process:

1. parses all JSON and JSONL records and applies deterministic, typed redaction
   markers for recognized sensitive-value shapes;
2. removes upstream session identifiers and replaces them with sequential,
   release-local identifiers such as `wildchat-sanitized-001`;
3. performs a record-by-record review for contact details, network and physical
   addresses, account identifiers, identifying personal names, credentials and
   cookies, sexual content involving minors, and actionable malware or account
   compromise;
4. excludes records that cannot be made safe while preserving research value;
   and
5. validates JSONL structure, character counts, cache alignment, and release
   scanner results.

Release-local identifiers are sequential labels, not hashes or transformations
of upstream identifiers. No reverse mapping or original WildChat session
identifier is published in this repository. Credential-shaped values were
handled as text patterns only; no credential, cookie, token, or account value
was tested against a provider or validated over a network.

## Fields

Each JSONL row contains:

- `record_id`: sequential release-local identifier;
- `source`: historical evaluation classification retained from the paper-time
  artifact;
- `char_count`: Unicode character count of the released transcript; and
- `transcript`: role-labelled conversation text, including typed redactions
  where necessary.

## Sensitive-content warning

The subset contains real-world conversation material and may be distressing or
harmful. Topics can include manipulation, self-harm, malware, credential theft,
sexual material, discrimination, or other unsafe behavior. Code, commands,
URLs, and instructions inside transcripts are inert research examples: do not
execute them, treat them as operational guidance, or assume they are accurate.

Automated pattern scanning and manual review reduce risk but cannot guarantee
that every sensitive fact, indirect identifier, or harmful implication has been
removed. Users remain responsible for appropriate access controls, review, and
compliance with applicable law and provider terms.

## External model-provider processing

Running the realism evaluation transmits the supplied transcript text to the
configured model providers for classification or judging. Before running it,
review each provider's current terms, privacy policy, retention settings,
regional processing, and data-use controls. Do not run the evaluation with data
you are not authorized to send to those providers.

Reading or inspecting the released JSONL locally does not itself contact a
model provider. The repository's release-safety checks also perform no provider
validation or network lookup.

## Takedown and correction requests

For a privacy, safety, attribution, or takedown request, use the repository's
private GitHub Security Advisory reporting channel when available. Do not put
sensitive personal data or a suspected credential in a public issue. If the
private channel is unavailable, open a minimal repository issue asking the
maintainers for a private contact route and include only the release-local
`record_id` plus a non-sensitive summary.

Maintainers should investigate the released artifact, remove or correct the
affected record where warranted, reindex release-local IDs and cached topic
scores, rerun the release checks, and document the change without republishing
sensitive material.

## Known limitations

- The 19 records are a small, historically selected evaluation subset and are
  not suitable for population-level conclusions.
- Safety review is contextual and pattern scanners have both false positives
  and false negatives.
- Redactions and one exclusion mean public reruns are not bit-for-bit identical
  to the paper-time 20-record realism benchmark.
- The upstream dataset and its terms can change; users should consult the
  current WildChat-1M dataset card and ODC-By text before redistribution.
