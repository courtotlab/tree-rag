# Double-blind artifact

The private development repository contains author metadata and Git history.
Do not submit it directly for anonymous review.

Create a separate, new artifact directory:

    scripts/make_anonymous_artifact.sh /absolute/new/path/treequest-anonymous

The builder is non-destructive. It refuses an existing destination, copies only
release-eligible files, excludes Git history, environments, caches, generated
corpora, trees, questions, reports, results, and logs, and replaces package and
citation metadata with anonymous templates. It then creates a checksum manifest.

Before uploading the artifact, maintainers must inspect it for:

- names, email addresses, affiliations, usernames, and lab/repository URLs;
- absolute local paths and hostnames;
- acknowledgements, grant identifiers, and self-identifying issue links;
- private corpus or benchmark material;
- generated answers, traces, prompts, logs, caches, and model endpoint secrets;
- Git history or timestamps that reveal identity;
- nonanonymous PDF metadata.

The anonymous artifact and the post-acceptance public repository are different
deliverables. Restore author metadata only after the review policy permits it.
