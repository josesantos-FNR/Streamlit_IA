import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. DICIONÁRIOS DE MAPEAMENTO (Interface -> IA)
# ==========================================
# Aqui invertemos um pouco a lógica para a interface ficar bonita: 
# A chave é o texto que o usuário vê, o valor é o número que a IA entende.

opcoes_renda = {
    'Não Declarada / Incompleta': 0,
    '0 < Renda <= 0,5 salário mínimo': 1,
    '0,5 < Renda <= 1,0 salário mínimo': 2,
    '1,0 < Renda <= 1,5 salário mínimo': 3,
    '1,5 < Renda <= 2,5 salários mínimos': 4,
    '2,5 < Renda <= 3,5 salários mínimos': 5,
    'Renda > 3,5 salários mínimos': 6
}

opcoes_financiamento = {
    'Sem Programa Associado': 0,
    'UAB': 1,
    'Outros Recursos Externos': 2,
    'MedioTec': 3,
    'Bolsa Formação': 4, 
    'E-TEC': 5,
    'Programa EJA INTEGRADA - EPT (SEB/MEC)': 6
}

opcoes_modalidade = {
    'Educação Presencial': 0,
    'Educação a Distância': 1
}

opcoes_turno = {
    'Matutino': 1,
    'Vespertino': 2,
    'Noturno': 3,
    'Integral': 4
}


opcoes_uf = {
    'AC': 0, 'AL': 1, 'AM': 2, 'AP': 3, 'BA': 4, 'CE': 5, 'DF': 6, 'ES': 7,
    'GO': 8, 'MA': 9, 'MG': 10, 'MS': 11, 'MT': 12, 'PA': 13, 'PB': 14,
    'PE': 15, 'PI': 16, 'PR': 17, 'RJ': 18, 'RN': 19, 'RO': 20, 'RR': 21,
    'RS': 22, 'SC': 23, 'SE': 24, 'SP': 25, 'TO': 26
}

# ==========================================
# 2. CONFIGURAÇÃO DA PÁGINA E MODELO
# ==========================================
st.set_page_config(page_title="Previsão de Evasão Escolar", layout="centered")
st.title("🎓 Previsão de Risco de Evasão Escolar")
st.write("Insira os dados do aluno abaixo para calcular a probabilidade de abandono.")

@st.cache_resource
def carregar_modelo():
    return joblib.load('modelo_rf_evasao.pkl')

modelo = carregar_modelo()

# ==========================================
# 3. INTERFACE DE ENTRADA DE DADOS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    carga_horaria = st.number_input("Carga Horária", min_value=0)
    fator_esforco = st.number_input("Fator Esforço Curso", min_value=0.0, format="%.2f")
    idade = st.number_input("Idade", min_value=0, max_value=120)
    
    # Selectbox mostra as chaves (textos)
    renda_selecionada = st.selectbox("Renda Familiar", list(opcoes_renda.keys()))

with col2:
    financiamento_selecionado = st.selectbox("Fonte de Financiamento", list(opcoes_financiamento.keys()))
    modalidade_selecionada = st.selectbox("Modalidade de Ensino", list(opcoes_modalidade.keys()))
    turno_selecionado = st.selectbox("Turno", list(opcoes_turno.keys()))
    
    # Nova caixa de seleção da UF
    uf_selecionada = st.selectbox("Estado (UF)", list(opcoes_uf.keys()))

# ==========================================
# 4. BOTÃO DE PREVISÃO
# ==========================================
if st.button("📊 Calcular Risco de Evasão"):
    
    # Traduzindo os textos selecionados para os números
    renda_num = opcoes_renda[renda_selecionada]
    financiamento_num = opcoes_financiamento[financiamento_selecionado]
    modalidade_num = opcoes_modalidade[modalidade_selecionada]
    turno_num = opcoes_turno[turno_selecionado]
    uf_num = opcoes_uf[uf_selecionada] # <-- Traduzindo a UF
    
    # Colocando a variável uf_num no DataFrame (agora é tudo número!)
    dados_entrada = pd.DataFrame([[
        carga_horaria, fator_esforco, financiamento_num, 
        idade, modalidade_num, renda_num, turno_num, uf_num
    ]], columns=[
        'Carga Horaria', 'Fator Esforço Curso', 'Fonte de Financiamento', 
        'Idade', 'Modalidade de Ensino', 'Renda Familiar', 'Turno', 'UF'
    ])
    
    # O seu mapa diz que Abandono/Cancelada é a classe 0.
    # predict_proba retorna [prob_classe_0, prob_classe_1]. Queremos a probabilidade da 0.
    probabilidade_evasao = modelo.predict_proba(dados_entrada)[0][0] 
    
    st.markdown("---")
    st.subheader("Resultado da Previsão")
    
    porcentagem = probabilidade_evasao * 100
    
    if porcentagem > 50:
        st.error(f"⚠️ **ALTO RISCO!** A probabilidade de abandono (evasão) deste aluno é de **{porcentagem:.1f}%**.")
    else:
        st.success(f"✅ **BAIXO RISCO!** A probabilidade de abandono deste aluno é de apenas **{porcentagem:.1f}%**.")
