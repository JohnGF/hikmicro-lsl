SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
DEVICE ?= /dev/video0
RECORDINGS ?= recordings
FILE ?= $(shell ls -t $(RECORDINGS)/*.bin 2>/dev/null | head -n1)

.PHONY: setup test lint stream replay clean-recordings

setup:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

test:
	uv run pytest

lint:
	uv run ruff check

stream:
	uv run hikmicro-lsl stream --device $(DEVICE) --raw

replay:
	uv run hikmicro-lsl replay $(FILE)

clean-recordings:
	rm -f $(RECORDINGS)/*.bin $(RECORDINGS)/*.jsonl $(RECORDINGS)/frames.latest
