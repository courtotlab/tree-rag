# Security

Please report security or accidental-data-disclosure issues privately to the
maintainers rather than opening a public issue. Never attach corpus files,
generated trees, prompts containing confidential passages, or benchmark outputs.

Before pushing, run:

```bash
git ls-files | grep -E 'corpus_tree|benchmark_report|questions|qms_answers|folders/'
```

The command should print nothing except source files whose names describe the
public experiment.

