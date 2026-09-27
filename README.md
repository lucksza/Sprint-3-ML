# Ford × FIAP 2026 — Desafio 02: Impulsionando o VIN Share · Sprint 3 · Inteligência Artificial & Machine Learning

**Integrantes:**
- Lucca Borges — RM 554608
- Ruan Melo — RM 557599
- Rodrigo Jimenez — RM 558148
- João Victor Franco — RM 556790
- Bruno Leão — RM 555563

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c1M1YZlpftawTe-lVuRdlThBAp8ULrn0?usp=sharing)

- **Notebook no Google Colab:** https://colab.research.google.com/drive/1c1M1YZlpftawTe-lVuRdlThBAp8ULrn0?usp=sharing
- **Repositório:** https://github.com/lucksza/Sprint-3-ML

## Solução

Modelo de Machine Learning que, **no momento da venda**, estima a probabilidade de o cliente **sair da rede Ford nos 24 meses seguintes** (`churn_rede_24m`) e o classifica em faixas de risco (Alto / Médio / Baixo) com uma ação sugerida. Assim, a concessionária recebe **leads de pós-venda priorizados**.

- **Problema:** classificação binária supervisionada, classes desbalanceadas (14,2% de churn).
- **Modelo selecionado:** XGBoost ajustado (`RandomizedSearchCV`), comparado com Regressão Logística, Árvore de Decisão, Random Forest, XGBoost padrão e Rede Neural (MLP).
- **Resultado no teste (100 mil registros não usados no ajuste dos estimadores):** PR-AUC 0,301 (acaso = 0,142), ROC-AUC 0,738. Os 20 % de maior risco concentram 43,7 % dos churners (lift 2,19×).
- **Preditores restritos às informações disponíveis na compra:** o modelo usa só as 22 variáveis do layout do arquivo operacional.
- **Deploy:** o artefato de inferência (`modelo/modelo_churn_rede_ford.joblib` + `ford_churn_pipeline.py`) é carregado e validado em processo Python separado, a partir do CSV operacional cru.

## Entregáveis pedidos → arquivos

| Pedido no enunciado | Arquivo |
|---|---|
| Notebook ou código-fonte com as etapas desenvolvidas | `Ford_Desafio2_IA_ML.ipynb` (executado, com saídas) · `Ford_Desafio2_IA_ML.html` (mesma versão, abre no navegador) |
| Documentação resumida da solução | `Documentacao_Resumida_IA_ML.pdf` |
| Comparação dos modelos e das métricas utilizadas | notebook, seções 3.3–3.5 e 4.1–4.4 · PDF, seção 4 |
| Conclusão com o modelo selecionado e justificativa | notebook, seção 5 · PDF, seção 5 |

## Estrutura

```
├── Ford_Desafio2_IA_ML.ipynb        notebook completo (compreensão → preparação → modelos → avaliação → conclusão)
├── Ford_Desafio2_IA_ML.html         versão HTML do notebook executado
├── Documentacao_Resumida_IA_ML.pdf  documentação resumida
├── requirements.txt                 bibliotecas e versões usadas
├── figuras/                         gráficos gerados pelo notebook + tabelas de métricas + resumo_resultados.json
├── ford_churn_pipeline.py           módulo de pré-processamento (gerado pelo notebook; usado no treino e na inferência)
├── modelo/modelo_churn_rede_ford.joblib   artefato de inferência (pipeline completo + cortes das faixas)
├── modelo/validar_artefato.py      valida o artefato em processo Python separado (gerado pelo notebook)
└── dados/                           coloque aqui os CSVs fornecidos (não incluídos pelo tamanho)
```

## Dados

- `ford_clientes_historico_completo.csv`: **base de modelagem** (500 mil clientes, 37 colunas, contém o alvo).
- `ford_clientes_operacional_compra.csv`: recorte exato do histórico (mesmos clientes, 24 colunas da compra, **sem alvo**). Não é usado no treino. Define as variáveis disponíveis na venda e é o layout de entrada da simulação de deploy.

## Como executar

**Localmente**
1. Python 3.11+ e `pip install -r requirements.txt`
2. Copie os dois CSVs para a pasta `dados/` (o notebook também procura na própria pasta e na pasta acima).
3. Abra `Ford_Desafio2_IA_ML.ipynb` e execute todas as células (~12 min em uma máquina de 16 núcleos; semente fixa = resultados reproduzíveis).

**No Google Colab** ([link](https://colab.research.google.com/drive/1c1M1YZlpftawTe-lVuRdlThBAp8ULrn0?usp=sharing))
1. Coloque os dois CSVs em uma pasta do Google Drive (ex.: `Sprint-ML3`).
2. Na primeira célula, monte o Drive e copie os arquivos para `/content`:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp /content/drive/MyDrive/Sprint-ML3/ford_clientes_*.csv /content/
   !pip install -q shap
   ```
3. *Ambiente de execução → Executar tudo.* No Colab gratuito (2 núcleos) a execução completa é mais demorada, e versões diferentes das bibliotecas podem mudar as últimas casas decimais dos resultados.
