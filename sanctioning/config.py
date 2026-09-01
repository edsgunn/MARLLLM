"""Config loading. One YAML drives the whole pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class RunSpec:
    run_id: str          # unique per seed; used as the run_id on every Post
    substrate: str       # behaviour family label
    path: str            # directory containing traces/ckpt_*/traces.json


@dataclass
class JudgeCfg:
    provider: str = "anthropic"        # "anthropic" | "openai"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 1024
    window_k: int = 6
    concurrency: int = 4               # keep low by default
    max_retries: int = 4               # SDK-level retry budget per call
    json_mode: bool = False            # OpenAI json_object mode; Anthropic ignores


@dataclass
class EmbeddingCfg:
    # "openai" needs OPENAI_API_KEY; "tfidf" is a no-API local fallback (sklearn).
    provider: str = "tfidf"
    model: str = "text-embedding-3-small"


@dataclass
class ClusterCfg:
    method: str = "agglomerative"      # "agglomerative" | "hdbscan"
    # agglomerative:
    distance_threshold: float = 1.0    # cosine distance; clusters merged below this
    # hdbscan / general:
    min_cluster_size: int = 8
    # sensitivity sweep emitted alongside the main result:
    sweep: list = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2])


@dataclass
class ValidationCfg:
    presence_f1_min: float = 0.65
    valence_kappa_min: float = 0.6
    n_label_samples: int = 200


@dataclass
class Paths:
    traces_root: str = "runs/cultural_emergence"
    output_dir: str = "runs/sanctioning"
    # everything below is derived from output_dir unless overridden
    events: Optional[str] = None
    clusters: Optional[str] = None
    figures: Optional[str] = None
    cache_db: Optional[str] = None
    labels: Optional[str] = None
    validation: Optional[str] = None

    def resolve(self):
        self.events = self.events or os.path.join(self.output_dir, "events.jsonl")
        self.clusters = self.clusters or os.path.join(self.output_dir, "clusters.jsonl")
        self.figures = self.figures or os.path.join(self.output_dir, "figures")
        self.cache_db = self.cache_db or os.path.join(self.output_dir, "cache.sqlite")
        self.labels = self.labels or os.path.join(self.output_dir, "labels.jsonl")
        self.validation = self.validation or os.path.join(self.output_dir, "validation")
        return self


@dataclass
class Config:
    judge: JudgeCfg
    embedding: EmbeddingCfg
    clustering: ClusterCfg
    validation: ValidationCfg
    paths: Paths
    corpus: list  # list[RunSpec]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        judge = JudgeCfg(**(raw.get("judge") or {}))
        emb = EmbeddingCfg(**(raw.get("embedding") or {}))
        clus = ClusterCfg(**(raw.get("clustering") or {}))
        val = ValidationCfg(**(raw.get("validation") or {}))
        paths = Paths(**(raw.get("paths") or {})).resolve()
        corpus = [RunSpec(**r) for r in (raw.get("corpus") or [])]
        return Config(judge, emb, clus, val, paths, corpus)

    def make_dirs(self):
        os.makedirs(self.paths.output_dir, exist_ok=True)
        os.makedirs(self.paths.figures, exist_ok=True)
        os.makedirs(self.paths.validation, exist_ok=True)
