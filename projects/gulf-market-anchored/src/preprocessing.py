from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

@dataclass
class Preprocessor:
    scale: bool
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    def fit(self,X):
        X=np.asarray(X,float); self.imputer=SimpleImputer(strategy="median").fit(X); Z=self.imputer.transform(X)
        if self.scale: self.scaler=StandardScaler().fit(Z)
        return self
    def transform(self,X):
        if self.imputer is None: raise RuntimeError("preprocessor not fit")
        Z=self.imputer.transform(np.asarray(X,float))
        return self.scaler.transform(Z) if self.scaler is not None else Z
    def fit_transform(self,X): self.fit(X); return self.transform(X)
