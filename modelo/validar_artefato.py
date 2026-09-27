"""Valida o artefato de inferência num processo Python separado (sem o notebook).
Uso: python validar_artefato.py <pasta_do_modulo> <modelo.joblib> <csv_operacional> <ids.csv> <saida.csv>"""
import sys

import joblib
import pandas as pd

pasta_modulo, caminho_modelo, csv_operacional, csv_ids, csv_saida = sys.argv[1:6]
sys.path.insert(0, pasta_modulo)  # onde está ford_churn_pipeline.py
artefato = joblib.load(caminho_modelo)
ids = pd.read_csv(csv_ids)["cliente_id"]
operacional = pd.read_csv(csv_operacional)  # layout operacional cru, sem nenhum tratamento
lote = operacional[operacional["cliente_id"].isin(ids)]
prob = artefato["pipeline"].predict_proba(lote[artefato["features"]])[:, 1]
pd.DataFrame({"cliente_id": lote["cliente_id"].to_numpy(), "prob": prob}).to_csv(csv_saida, index=False, float_format="%.17g")
winsor = artefato["pipeline"].named_steps["prep"].named_transformers_["assim"].named_steps["winsor"]
print(f"processo separado OK | {len(lote)} clientes pontuados | Winsorizer carregado do módulo: {type(winsor).__module__}")
