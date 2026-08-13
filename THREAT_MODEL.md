# TreeRAG privacy and security threat model

Status: canonical-v1 research artifact  
Scope: document-tree construction, interactive retrieval, answer generation,
evaluation, and anonymous/public release

## 1. Security objective

TreeRAG should let an authorized user retrieve evidence from a governed
document hierarchy without moving corpus-derived text outside the approved
inference boundary or accidentally publishing it through code, logs,
benchmarks, artifacts, or manuscript materials.

This is a research-system threat model, not a claim of regulatory
certification, clinical safety, formal noninterference, or differential privacy.

## 2. Protected assets

The following are sensitive whenever they derive from a private deployment:

- source documents and extracted text;
- hierarchy structure, node names, summaries, and identifiers;
- user questions and query history;
- retrieved passages and evidence bundles;
- generated answers and citations;
- traversal decisions, traces, prompts, and judge inputs;
- caches, checkpoints, failure logs, and benchmark reports;
- model endpoint credentials, hostnames, and infrastructure metadata;
- aggregate statistics not yet approved for disclosure.

Hierarchy summaries are treated as corpus content. They may reveal facts even
when the corresponding source document is absent.

## 3. Trust boundaries

### 3.1 Private corpus boundary

Raw documents, parsed text, generated hierarchy material, retrieval traces, and
private outputs remain in the institutionally approved environment. Access
control, storage encryption, backup policy, endpoint logging, and retention are
deployment responsibilities.

### 3.2 Model inference boundary

Private inference is permitted only through an approved on-premises
open-weights model endpoint. Selected summaries and evidence passages cross the
process boundary to that endpoint, so the endpoint, transport, request logs,
model cache, and operators are inside the trusted computing base.

A loopback SSH forward protects transport routing but does not make a remote
endpoint trustworthy by itself. Endpoint identity and institutional controls
must be established separately.

### 3.3 Public evaluation boundary

The MultiHop-RAG evaluation uses a separate public dataset, hierarchy, indexes,
checkpoints, and result directory. Public and private caches must not be reused
across boundaries. A public result cannot be inferred from or silently mixed
with private question-level data.

### 3.4 Publication boundary

Only source code, public experiment material, public aggregate results, and
explicitly approved private aggregate statistics may cross into the anonymous
or public artifact. Private per-question values are excluded even if they omit
document text.

## 4. Threats and controls

| Threat | Example failure | Primary controls | Residual risk |
|---|---|---|---|
| Accidental source-control disclosure | A hierarchy, report, answer checkpoint, or environment file is committed. | Denylisted generated paths, separate public/private roots, non-destructive anonymous artifact builder, release scan, and human review. | Novel filenames or encoded archives can evade filename-based rules. |
| Misconfigured model endpoint | Private evidence is sent to a hosted or unapproved service. | Explicit endpoint configuration, on-prem deployment requirement, fail-closed health checks, and deployment review. | A trusted hostname can still route incorrectly; endpoint authentication and network policy remain external controls. |
| Prompt or trace logging | Requests containing passages are retained by Ollama, proxies, shell history, notebooks, or observability tools. | Disable or restrict payload logging, redact operational logs, protect checkpoints, and never publish model prose from private runs. | Administrators inside the trust boundary may still access retained logs. |
| Document prompt injection | A document instructs the model to reveal data, ignore search policy, or call an external tool. | Documents are presented as evidence rather than instructions; the standalone agent exposes no network or arbitrary tool execution; answer prompts restrict output to evidence-grounded answers. | Instruction/data separation is not formally guaranteed, and malicious text can still influence answers or traversal. |
| Cross-corpus cache contamination | A public run retrieves a private cached summary or embedding. | Separate roots, caches, indexes, manifests, and immutable run IDs; prohibit private artifacts in public analysis. | Operator error remains possible without filesystem-level isolation. |
| Citation or path leakage | An answer exposes internal filenames, folder names, or URLs. | Keep private answers inside the approved boundary; public release uses only public corpus identifiers; aggregate analyses omit identifiers. | Authorized private users still see evidence identifiers by design. |
| Overbroad result release | Per-question scores appear harmless but reveal workload or corpus properties. | Private analysis emits aggregate statistics only and excludes qids from released outputs. | Small or highly stratified aggregates can still disclose information; disclosure approval is required. |
| Membership or attribute inference | Repeated questions reveal whether a sensitive policy or fact exists. | Authentication, authorization, audit policy, rate limits, and corpus-level access controls are required around the host application. | TreeRAG itself does not implement document-level authorization or differential privacy. |
| Model hallucination or unsupported synthesis | The generated answer is treated as authoritative despite incomplete evidence. | Sufficiency checks, evidence citations, bounded not-found behavior, uncertainty-aware evaluation, and explicit limitations. | Citations do not prove the synthesis is correct; human review remains necessary for consequential use. |
| Denial of service or runaway cost | Wide hierarchy nodes or slow requests consume excessive inference time. | Canonical fanout cap, exact successful context-call cap, node/file/evidence limits, bounded retries, and cooperative wall-time checks. | An in-flight model call is not preempted and can exceed the nominal wall-clock allowance. |
| Re-identification through metadata | Authors, affiliations, usernames, local paths, Git history, or PDF metadata break double blindness. | Separate anonymous artifact, anonymous package metadata, excluded Git history, marker scan, checksum manifest, and manual PDF inspection. | Automated scans are incomplete and require human review. |

## 5. Prompt-injection assumptions

TreeRAG treats corpus text as untrusted data. The language model should not
interpret document instructions as authorization to alter budgets, access
external systems, disclose unrelated evidence, or override the user question.

The standalone release does not give the model shell, network, filesystem-write,
email, or arbitrary retrieval tools. It can request only the fixed retrieval
operations implemented by the host search state machine. This limits impact but
does not eliminate semantic prompt injection.

A production host should additionally enforce:

- document-level authorization before a node can enter the hierarchy;
- output filtering appropriate to the corpus and user role;
- endpoint egress restrictions;
- immutable audit events that exclude passage text where possible;
- incident response for accidental output disclosure;
- adversarial prompt-injection testing on synthetic documents.

## 6. Data lifecycle

### 6.1 Construction

Document parsing and summarization can expose entire source documents to the
configured model. Builder caches are sensitive and inherit the source corpus
retention and access policy. Construction is explicit and never triggered by a
query.

### 6.2 Query execution

A query can expose the question, selected node summaries, and selected evidence
to the model endpoint. Bounded search reduces the amount presented but is not a
privacy guarantee. The answer and citations remain sensitive outputs.

### 6.3 Evaluation

Private evaluation may process frozen questions and evidence inside the
approved environment. Released products contain only approved aggregate
numbers. Human-evaluation handling is controlled by the study team and is not
performed by this artifact.

### 6.4 Anonymous and public release

The anonymous artifact excludes identities, Git history, generated data,
outputs, and environment files. Post-acceptance publication restores author
metadata only when venue policy permits it. Public benchmark outputs require a
license and provenance review before release.

## 7. Regulated-setting claim boundary

The deployment motivates requirements such as local inference, provenance,
bounded operation, and auditable evidence paths. It does not establish that
TreeRAG is a medical device, validated clinical decision-support system, or
compliant with a particular regulation. The system is not evaluated for
diagnosis, treatment, patient outcome, or autonomous clinical action.

Any consequential use requires independent governance, validation,
authorization, monitoring, and human review appropriate to that use.

## 8. Residual limitations

- The model can reproduce sensitive text to an authorized questioner.
- Corpus-level authorization does not imply document-level authorization.
- Summaries and embeddings can be sensitive even without raw documents.
- Open-weight, on-premises inference improves control but does not guarantee
  confidentiality.
- Prompt injection and hallucination remain open risks.
- Cooperative time limits cannot cancel an in-flight request.
- Aggregate disclosure can still be unsafe for small groups.
- No formal privacy, security, or regulatory certification is claimed.
