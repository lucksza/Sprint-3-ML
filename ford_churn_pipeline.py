"""Pré-processamento do modelo de churn da rede Ford (Desafio 02). Usado no treino (notebook) e na produção (API)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def corrigir_inconsistencias(df):
    """Regra determinística (linha a linha): financiamento com prazo 0 → prazo e prestação ausentes."""
    d = df.copy()
    financiado_sem_prazo = (d["compra_a_vista"] == 0) & (d["prazo_financiamento_meses"] == 0)
    d.loc[financiado_sem_prazo, ["prazo_financiamento_meses", "prestacao_renda_ratio"]] = np.nan
    return d


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita cada coluna entre os quantis q_inf e q_sup aprendidos no treino (tratamento de outliers)."""

    def __init__(self, q_inf=0.005, q_sup=0.995):
        self.q_inf = q_inf
        self.q_sup = q_sup

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limite_inf_ = X.quantile(self.q_inf).to_numpy()
        self.limite_sup_ = X.quantile(self.q_sup).to_numpy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inf_, self.limite_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
