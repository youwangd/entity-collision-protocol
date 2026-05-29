# Engram reproducibility container.
# Pinned to Python 3.11 — matches the version used to produce every headline
# number in paper/40_results.md. See paper/REPRODUCIBILITY.md.
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: build toolchain for sqlite-vec / sentence-transformers wheels,
# git for any VCS-pinned deps, sqlite for FTS5, curl for the spaCy model pull.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        sqlite3 \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /engram

# Copy metadata first to maximize layer cache hits on dep installs.
COPY pyproject.toml README.md ./
COPY src ./src

# Install with all optional extras + dev. The [all] extra is defined in
# pyproject.toml as engram[llm,embeddings,vector,encryption,entity-ner].
RUN pip install -e '.[all,dev]' \
    && pip install hypothesis pytest-xdist \
    && python -m spacy download en_core_web_sm

# Now copy the rest. Keeps the slow pip layer cached when only paper/, evals/,
# tests/, scripts/ change.
COPY . .

# Sanity check at build time — fails the image if the test suite is red.
# Skip slow/chaos/mega_scale; those are run explicitly by reproduce.sh.
RUN python -m pytest -q --tb=line -m "not slow and not chaos and not mega_scale" \
    || (echo "ERROR: test suite red on container build" && exit 1)

CMD ["bash", "scripts/reproduce.sh"]
