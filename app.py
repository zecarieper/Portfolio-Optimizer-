import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador Ágora", layout="wide")
st.title("Simulador de Portfólio Interativo")
st.markdown("---")

# --- FUNÇÃO DE LEITURA (Para carregar o arquivo inicial) ---
def carregar_csv_inicial(arquivo):
    try:
        # Tenta ler com separador automático
        df = pd.read_csv(arquivo, sep=None, engine='python', index_col=0)
        # Limpeza de texto
        df = df.astype(str).replace({'%': '', 'R\$': '', ' ': ''}, regex=True)
        df = df.apply(lambda x: x.str.replace(',', '.') if x.dtype == "object" else x)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all').dropna(how='all', axis=1)

        # Ajuste de escala (se > 1, divide por 100)
        if df.mean().mean() > 1:
            df = df / 100
        return df
    except:
        return None

# --- BARRA LATERAL ---
st.sidebar.header("1. Upload Inicial")
st.sidebar.info("Carregue os CSVs para preencher a tabela.")
f_rent = st.sidebar.file_uploader("Rentabilidade.csv", type="csv")
f_vol = st.sidebar.file_uploader("Volatilidade.csv", type="csv")

st.sidebar.markdown("---")
rf = st.sidebar.number_input("Taxa Livre de Risco:", value=0.02, step=0.005, format="%.2f")

# --- LÓGICA PRINCIPAL ---
if f_rent and f_vol:
    # 1. Carrega os dados brutos do arquivo
    df_ret_raw = carregar_csv_inicial(f_rent)
    df_vol_raw = carregar_csv_inicial(f_vol)

    if df_ret_raw is not None and df_vol_raw is not None:

        # 2. MOSTRA TABELAS EDITÁVEIS
        st.subheader("Edite os Cenários Abaixo (Clique e digite)")

        col_edit1, col_edit2 = st.columns(2)

        with col_edit1:
            st.markdown("**Rentabilidade Esperada**")
            # O usuário edita e 'df_ret' recebe os novos valores
            df_ret = st.data_editor(df_ret_raw, height=200, key="edit_ret")

        with col_edit2:
            st.markdown("**Volatilidade (Risco)**")
            df_vol = st.data_editor(df_vol_raw, height=200, key="edit_vol")

        # 3. CÁLCULOS (Usando os dados JÁ editados)
        ativos = df_ret.index.intersection(df_vol.index)

        if len(ativos) < 2:
            st.error("Erro: Os nomes dos ativos não batem nas duas tabelas.")
        else:
            # Seleção
            selecao = st.multiselect("Selecione Ativos para Carteira:", ativos, default=ativos[:5])

            if len(selecao) >= 2:
                d_ret = df_ret.loc[selecao].T
                d_vol = df_vol.loc[selecao].T

                # Tratamento de zeros
                d_vol = d_vol.replace(0, 0.0001)

                mu = d_ret.mean()
                cov = d_ret.corr().multiply(d_vol.mean(), axis=0).multiply(d_vol.mean(), axis=1)

                n = len(selecao)

                # Otimização Markowitz
                def get_stats(w):
                    r = np.dot(w, mu)
                    v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                    return r, v, (r-rf)/v if v>0 else 0

                cons = ({'type':'eq', 'fun': lambda x: sum(x)-1})
                bounds = [(0,1)]*n

                try:
                    opt = minimize(lambda w: -get_stats(w)[2], [1/n]*n, bounds=bounds, constraints=cons)
                    r_opt, v_opt, s_opt = get_stats(opt.x)

                    # --- RESULTADOS ---
                    st.markdown("---")
                    st.subheader("Resultados")

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Gráfico
                        w_sim = np.random.dirichlet(np.ones(n), 2000)
                        r_sim = np.dot(w_sim, mu)
                        v_sim = np.sqrt(np.diag(np.dot(np.dot(w_sim, cov), w_sim.T)))

                        fig, ax = plt.subplots(figsize=(10,5))
                        sc = ax.scatter(v_sim, r_sim, c=(r_sim-rf)/v_sim, cmap='viridis', s=15, alpha=0.7)
                        plt.colorbar(sc, label="Sharpe")
                        ax.scatter(v_opt, r_opt, c='red', marker='*', s=200, label="Carteira eficiente")
                        ax.legend()
                        ax.set_xlabel("Risco")
                        ax.set_ylabel("Retorno")
                        st.pyplot(fig)

                    with col2:
                        # Tabela Final
                        st.metric("Sharpe Máximo", f"{s_opt:.2f}")
                        res = pd.DataFrame({'Ativo': selecao, 'Peso': opt.x})
                        res['Peso'] = res.Peso.apply(lambda x: f"{x:.1%}")
                        st.dataframe(res.sort_values('Peso', ascending=False), hide_index=True)

                except Exception:
                    st.warning("Cálculo inválido. Verifique os valores inseridos.")
    else:
        st.error("Erro ao ler CSVs.")
else:
    st.info("Faça o upload inicial para liberar a edição.")
