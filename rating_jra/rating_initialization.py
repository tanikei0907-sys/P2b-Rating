from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InitialRule:
    b: np.ndarray
    var0_d: np.ndarray
    beta: np.ndarray

    @classmethod
    def p0_like(cls, config_count: int, var0: float = 2.0) -> "InitialRule":
        return cls(
            b=np.zeros((config_count, 3), dtype=float),
            var0_d=np.full((config_count, 3), var0, dtype=float),
            beta=np.zeros((config_count, 4), dtype=float),
        )

    def validate(self, config_count: int) -> None:
        if self.b.shape != (config_count, 3):
            raise ValueError("bの形状が不正です")
        if self.var0_d.shape != (config_count, 3):
            raise ValueError("var0_dの形状が不正です")
        if self.beta.shape != (config_count, 4):
            raise ValueError("betaの形状が不正です")
        if not np.all(np.isfinite(self.b)) or not np.all(np.isfinite(self.beta)):
            raise ValueError("初期化係数が非有限です")
        if not np.all(np.isfinite(self.var0_d)) or np.any(self.var0_d <= 0):
            raise ValueError("初期分散は正の有限値である必要があります")


@dataclass
class FeatureBlock:
    X: np.ndarray
    N_sire: np.ndarray
    N_bms: np.ndarray
    sire_status: np.ndarray
    bms_status: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "FeatureBlock":
        return cls(
            X=np.full((n, 4), np.nan, dtype=float),
            N_sire=np.zeros(n, dtype=np.int32),
            N_bms=np.zeros(n, dtype=np.int32),
            sire_status=np.zeros(n, dtype=np.int8),
            bms_status=np.zeros(n, dtype=np.int8),
        )

    def validate(self, n: int) -> None:
        if self.X.shape != (n, 4):
            raise ValueError("血統特徴量の形状が不正です")
        for values in [self.N_sire, self.N_bms, self.sire_status, self.bms_status]:
            if len(values) != n:
                raise ValueError("血統補助列の行数が不正です")
