FROM python:3.11-slim

# uv for fast, reproducible installs from the lockfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first so this layer is cached when only source changes
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache \
        numpy \
        h5py \
        pyyaml

# Install the package itself (no deps re-download)
COPY src/ src/
RUN uv pip install --system --no-cache --no-deps .

# Data and config are mounted at runtime — nothing baked in
WORKDIR /data

ENTRYPOINT ["adapt-io"]
CMD ["--help"]
