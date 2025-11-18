# extratos_plantas_db.py

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import math
import unicodedata
import plotly.io as pio
pio.renderers.default = "notebook"

def remover_acentos(texto):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    return texto

def normalizar_colunas(df):
    df.columns = [remover_acentos(col) for col in df.columns]
    return df

def carregar_dados(caminho):
    df = pd.read_excel(caminho, sheet_name="Sheet1", engine="odf")
    return normalizar_colunas(df)

def buscar_por_apf(df, codigo_apf):
    return df[df['Registro da amostra APF'] == codigo_apf]

def listar_codigos_apf(df):
    return df['Registro da amostra APF'].dropna().unique()

def mostrar_dados_exemplo(caminho):
    df = carregar_dados(caminho)
    print("Total de registros carregados:", len(df))
    codigos_apf = listar_codigos_apf(df)
    print("Códigos de extrato (APF) encontrados:", codigos_apf[:10])
    if len(codigos_apf) > 0:
        codigo_exemplo = codigos_apf[0]
        print(f"\nDados para o código APF {codigo_exemplo}:")
        print(buscar_por_apf(df, codigo_exemplo))

def visualizar_apf_completo(df, codigo_apf):
    dados_apf = buscar_por_apf(df, codigo_apf)
    if dados_apf.empty:
        print(f"Nenhum dado encontrado para o código APF {codigo_apf}.")
    else:
        print(f"Dados completos para o código APF {codigo_apf}:")
        display(dados_apf)

def contar_amostras_por_familia(df):
    return df.dropna(subset=['Registro da amostra APF']).groupby('Família')['Registro da amostra APF'].nunique().sort_values(ascending=False)

def tabela_por_familia(df):
    tab = (
        df.dropna(subset=['Família'])
          .groupby('Família')
          .size()
          .reset_index(name='N_amostras')
          .sort_values('N_amostras', ascending=False)
    )
    return tab

def tabela_por_genero(df):
    tab = (
        df.dropna(subset=['Gênero'])
          .groupby('Gênero')
          .size()
          .reset_index(name='N_amostras')
          .sort_values('N_amostras', ascending=False)
    )
    return tab

def tabela_familia_genero(df):
    df_clean = df.dropna(subset=['Família', 'Gênero'])
    tab = pd.crosstab(df_clean['Família'], df_clean['Gênero'])
    return tab

import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_familia_genero(df):
    tab = c2b.tabela_familia_genero(df)
    plt.figure(figsize=(20,12))
    sns.heatmap(tab, cmap="viridis", norm=None)
    plt.title("Distribuição Família × Gênero")
    plt.tight_layout()
    plt.show()

def familias_com_mais_generos(df, top_n=10):
    tab = c2b.tabela_familia_genero(df)
    richness = tab.astype(bool).sum(axis=1).sort_values(ascending=False)
    return richness.head(top_n)



def exportar_dados_apf(df, codigo_apf, caminho_saida):
    dados = buscar_por_apf(df, codigo_apf)
    dados.to_csv(caminho_saida, index=False)
    print(f"Dados exportados para {caminho_saida}")

def filtrar_e_reorganizar_apf(df):
    df_filtrado = df.dropna(subset=['Registro da amostra APF'])
    colunas = list(df_filtrado.columns)
    if 'Registro da amostra APF' in colunas:
        colunas.remove('Registro da amostra APF')
        colunas = ['Registro da amostra APF'] + colunas
        df_filtrado = df_filtrado[colunas]
    return df_filtrado

def distribuicao_por_familia(df):
    return df['Família'].value_counts().sort_values(ascending=False)

def distribuicao_por_genero_especie(df):
    return df['Espécies'].value_counts().sort_values(ascending=False)

def plot_bar_familia(df, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)

    # usar o mesmo filtro do sunburst
    df_plot = df.dropna(subset=['Família', 'Gênero', 'Espécies']).copy()
    contagem = df_plot['Família'].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12,6))
    contagem.plot(kind='bar', ax=ax)
    ax.set_title('Distribuição por Família (taxonomia completa)')
    ax.set_xlabel('Família')
    ax.set_ylabel('Número de Amostras')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barplot_familia.png"))

    html_fig = px.bar(
        contagem.reset_index(),
        x=contagem.reset_index().columns[0],
        y=contagem.reset_index().columns[1],
        title='Distribuição por Família (taxonomia completa)'
    )
    html_path = os.path.join(output_dir, "barplot_familia.html")
    pio.write_html(html_fig, html_path)
    print(f"Gráficos salvos em {output_dir}")


def plot_bar_genero(df, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    contagem = distribuicao_por_genero_especie(df)
    fig, ax = plt.subplots(figsize=(12,6))
    contagem.plot(kind='bar', ax=ax)
    ax.set_title('Distribuição por Gênero/Espécies')
    ax.set_xlabel('Espécies')
    ax.set_ylabel('Número de Amostras')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barplot_genero.png"))
    html_fig = px.bar(contagem.reset_index(), x=contagem.reset_index().columns[0], y=contagem.reset_index().columns[1], title='Distribuição por Gênero/Espécies')
    html_path = os.path.join(output_dir, "barplot_genero.html")
    pio.write_html(html_fig, html_path)
    print(f"Gráficos salvos em {output_dir}")

def plot_sunburst_familia_genero(df, output_dir="images"):
    import os
    import plotly.express as px

    os.makedirs(output_dir, exist_ok=True)

    # remove rows missing Família OR Gênero OR Espécies
    df_plot = df.dropna(subset=['Família', 'Gênero', 'Espécies']).copy()
    df_plot['count'] = 1

    fig = px.sunburst(
        df_plot,
        path=['Família', 'Gênero', 'Espécies'],
        values='count',
        title='Distribuição Sunburst por Família, Gênero e Espécie'
    )

    output_path = os.path.join(output_dir, "sunburst_familia_genero.html")
    fig.write_html(output_path)

    print(f"Gráfico salvo em {output_path}")


import os
import pandas as pd
from collections import defaultdict

import os
import pandas as pd
from collections import defaultdict

def _norm(s):
    import unicodedata
    return unicodedata.normalize('NFKD', s).encode('ASCII','ignore').decode('ASCII').strip().lower()

def _resolve_col(df, target):
    tgt = _norm(target)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    raise KeyError(f"Required column '{target}' not found. Available: {list(df.columns)}")


def criar_batches_hierarquico_fam_gen_esp(
    data: pd.DataFrame,
    output_path: str,
    samples_per_batch: int = 80,
    qc_samples=('Blank', 'QC_Inter_Batch', 'QC_Intra_Batch'),
    qc_structure=(3, 3, 2),
    batch_structure=(24, 24, 32),
    family_col='Familia',   # nomes normalizados
    genero_col='Genero',
    especie_col='Especies',
    random_state=None,      # se quiser embaralhar dentro de cada nível mantendo hierarquia
    keep_na_labels=True,    # trata NaN como rótulo 'NA' para não perder linhas
):
    """
    Monta lotes hierárquicos: Família -> Gênero -> Espécie.

    - Famílias ordenadas da mais populosa para a menos populosa.
    - Dentro de cada família: gêneros mais populosos primeiro, depois espécies.
    - TODO o dataframe é primeiro ordenado (Família > Gênero > Espécie),
      depois cortado sequencialmente em blocos de `samples_per_batch`.
    - Isso permite misturar famílias apenas nas fronteiras entre blocos,
      mas garante que cada família apareça em um bloco CONTÍNUO de batches
      (nunca em batch_51 e batch_86, por exemplo).

    QC são inseridos depois via montar_batch_com_qcs.
    """

    # -------- validações de parâmetros --------
    if sum(batch_structure) != samples_per_batch:
        raise ValueError("sum(batch_structure) must equal samples_per_batch.")
    if len(batch_structure) != len(qc_structure):
        raise ValueError("batch_structure and qc_structure must have the same length.")
    if not qc_samples:
        raise ValueError("qc_samples must contain at least one QC label.")

    fam_col = _resolve_col(data, family_col)
    gen_col = _resolve_col(data, genero_col)
    esp_col = _resolve_col(data, especie_col)

    os.makedirs(output_path, exist_ok=True)

    df = data.copy()

    # Opcional: embaralhar para quebrar viés de ordem original
    # (a ordenação hierárquica vem em seguida)
    if random_state is not None:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Padroniza rótulos NaN (opcional)
    if keep_na_labels:
        df[fam_col] = df[fam_col].fillna("NA")
        df[gen_col] = df[gen_col].fillna("NA")
        df[esp_col] = df[esp_col].fillna("NA")

    # ===== 1) ORDENAR FAMÍLIAS POR TAMANHO TOTAL (desc) =====
    fam_sizes = df.groupby(fam_col, dropna=False).size().sort_values(ascending=False)
    ordered_families = list(fam_sizes.index)

    # ===== 2) CONSTRUIR UMA ÚNICA TABELA ORDENADA GLOBALMENTE =====
    all_blocks = []

    for fam in ordered_families:
        fam_df = df[df[fam_col] == fam].copy()

        # 2a) Ordenar gêneros por tamanho dentro da família (desc)
        gen_sizes = (fam_df.groupby(gen_col, dropna=False)
                          .size().sort_values(ascending=False))
        ordered_genera = list(gen_sizes.index)

        ordered_rows = []
        for gen in ordered_genera:
            gen_df = fam_df[fam_df[gen_col] == gen].copy()

            # 2b) Dentro de cada gênero, ordenar espécies por tamanho (desc)
            esp_sizes = (gen_df.groupby(esp_col, dropna=False)
                               .size().sort_values(ascending=False))

            for esp in esp_sizes.index:
                block = gen_df[gen_df[esp_col] == esp]
                ordered_rows.append(block)

        fam_ordered = pd.concat(ordered_rows, ignore_index=True) if ordered_rows else fam_df

        # esse bloco da família (com gêneros/espécies organizados) vai para a lista global
        all_blocks.append(fam_ordered)

    # Concatena TUDO numa única tabela ordenada
    global_ordered = pd.concat(all_blocks, ignore_index=True)

    # ===== 3) CORTAR A TABELA ORDENADA EM LOTES SEQUENCIAIS =====
    batches = []
    batch_num = 1

    total = len(global_ordered)
    start = 0

    while start < total:
        end = min(start + samples_per_batch, total)
        chunk = global_ordered.iloc[start:end].copy()
        start = end

        final_batch = montar_batch_com_qcs(
            batch=chunk,
            batch_structure=batch_structure,
            qc_structure=qc_structure,
            qc_samples=list(qc_samples)
        )
        final_batch['Batch'] = f'batch_{batch_num}'
        batches.append(final_batch)
        batch_num += 1

    # ===== 4) SALVAR =====
    for df_b in batches:
        batch_name = df_b['Batch'].iloc[0]
        df_b.to_csv(os.path.join(output_path, f'{batch_name}.csv'), index=False)

    print(
        f"{len(batches)} batches hierárquicos criados em '{output_path}' "
        f"(Família > Gênero > Espécie; famílias em blocos contínuos; "
        f"pequenas famílias combinadas sequencialmente)."
    )
    return batches


def criar_batches_por_familia_genero(data, output_path, 
                                     samples_per_batch=80, 
                                     qc_samples=['Blank', 'QC_Inter_Batch', 'QC_Intra_Batch'],
                                     qc_structure=[3, 3, 2],
                                     batch_structure=[24, 24, 32],
                                     family_col='Família', genero_col='Gênero'):
    """
    Cria batches agrupando amostras preferencialmente por Família + Gênero, 
    com inserção de QCs conforme a estrutura definida.
    """

    if sum(batch_structure) != samples_per_batch:
        raise ValueError("A soma de batch_structure deve ser igual a samples_per_batch.")

    os.makedirs(output_path, exist_ok=True)
    batches = []
    batch_num = 1

    # Agrupa por Família + Gênero
    grouped = data.groupby([family_col, genero_col])
    group_sizes = grouped.size().sort_values(ascending=False)

    leftovers = pd.DataFrame()

    for (family, genero), _ in group_sizes.items():
        group = grouped.get_group((family, genero))
        
        while len(group) >= samples_per_batch:
            batch = group.iloc[:samples_per_batch]
            group = group.iloc[samples_per_batch:]
            final_batch = montar_batch_com_qcs(batch, batch_structure, qc_structure, qc_samples)
            final_batch['Batch'] = f'batch_{batch_num}'
            batches.append(final_batch)
            batch_num += 1

        # guarda sobras
        leftovers = pd.concat([leftovers, group])

    # Agrupa sobras por Família
    family_groups = leftovers.groupby(family_col)
    remaining = pd.DataFrame()

    for family, fam_group in family_groups:
        while len(fam_group) >= samples_per_batch:
            batch = fam_group.iloc[:samples_per_batch]
            fam_group = fam_group.iloc[samples_per_batch:]
            final_batch = montar_batch_com_qcs(batch, batch_structure, qc_structure, qc_samples)
            final_batch['Batch'] = f'batch_{batch_num}'
            batches.append(final_batch)
            batch_num += 1
        remaining = pd.concat([remaining, fam_group])

    # Junta todas as sobras para formar os últimos batches mistos
    while len(remaining) >= samples_per_batch:
        batch = remaining.iloc[:samples_per_batch]
        remaining = remaining.iloc[samples_per_batch:]
        final_batch = montar_batch_com_qcs(batch, batch_structure, qc_structure, qc_samples)
        final_batch['Batch'] = f'batch_{batch_num}'
        batches.append(final_batch)
        batch_num += 1

    # último batch incompleto (se houver)
    if not remaining.empty:
        final_batch = montar_batch_com_qcs(remaining, batch_structure, qc_structure, qc_samples)
        final_batch['Batch'] = f'batch_{batch_num}'
        batches.append(final_batch)

    # Salvar os batches
    for df in batches:
        batch_name = df['Batch'].iloc[0]
        df.to_csv(os.path.join(output_path, f'{batch_name}.csv'), index=False)

    print(f"{len(batches)} batches criados em '{output_path}' agrupando por Família e Gênero.")
    return batches


def montar_batch_com_qcs(batch, batch_structure, qc_structure, qc_samples):
    """
    Monta o DataFrame final de um batch com QCs intercalados.
    """
    final_batch = pd.DataFrame()
    start = 0

    for size, num_qcs in zip(batch_structure, qc_structure):
        # QCs intermediários
        qc_block = pd.DataFrame({
            'sampleid': qc_samples * num_qcs,
            'Tipo': ['QC'] * (len(qc_samples) * num_qcs)
        })
        final_batch = pd.concat([final_batch, qc_block], ignore_index=True)

        # Bloco de amostras reais
        sample_block = batch.iloc[start:start+size].copy()
        sample_block['Tipo'] = 'Amostra'
        final_batch = pd.concat([final_batch, sample_block], ignore_index=True)
        start += size

    # QCs finais
    qc_final = pd.DataFrame({
        'sampleid': qc_samples,
        'Tipo': ['QC'] * len(qc_samples)
    })
    final_batch = pd.concat([final_batch, qc_final], ignore_index=True)

    return final_batch

# Gera resumo da composição de famílias e gêneros por batch
def gerar_resumo_composicao(batches, output_path):
    """
    Gera resumo (Familia, Genero, Contagem, Batch) tolerante a:
    - acentos/variações de nome de coluna,
    - variações no valor 'Amostra' (Tipo),
    - NaN em taxonomia (preenche com 'NA').
    """
    import os
    import pandas as pd

    resumo = []
    problemas = []  # coletar batches com 100% NA nas amostras

    for df_b in batches:
        # resolver colunas presentes neste batch
        try:
            col_tipo = _resolve_col(df_b, 'Tipo')
        except KeyError:
            # se não houver Tipo, nada a fazer
            continue

        # filtro robusto de Amostra
        tipo_norm = df_b[col_tipo].astype(str).str.strip().str.lower()
        df_amostras = df_b[tipo_norm.eq('amostra')].copy()
        if df_amostras.empty:
            # sem amostras neste batch (só QC?) — pula
            continue

        # resolver nomes de família e gênero (aceita com/sem acento)
        try:
            col_fam = _resolve_col(df_amostras, 'Familia')
        except KeyError:
            # se não existir Família nas amostras, marcar problema
            problemas.append((df_b['Batch'].iloc[0], 'coluna Familia ausente nas amostras'))
            continue

        try:
            col_gen = _resolve_col(df_amostras, 'Genero')
        except KeyError:
            problemas.append((df_b['Batch'].iloc[0], 'coluna Genero ausente nas amostras'))
            continue

        # preencher NaN para não “sumir” no groupby e facilitar diagnóstico
        df_amostras[col_fam] = df_amostras[col_fam].fillna('NA')
        df_amostras[col_gen] = df_amostras[col_gen].fillna('NA')

        # checar se tudo ficou NA — indica origem sem taxonomia nas amostras
        if df_amostras[col_fam].eq('NA').all() and df_amostras[col_gen].eq('NA').all():
            problemas.append((df_b['Batch'].iloc[0], 'todas as amostras com taxonomia NA'))

        contagem = (df_amostras
                    .groupby([col_fam, col_gen], dropna=False)
                    .size()
                    .reset_index(name='Contagem'))

        contagem['Batch'] = df_b['Batch'].iloc[0]
        # padroniza cabeçalhos de saída
        contagem = contagem.rename(columns={col_fam: 'Família', col_gen: 'Gênero'})
        resumo.append(contagem)

    resumo_df = (pd.concat(resumo, ignore_index=True)
                 if resumo else pd.DataFrame(columns=['Família','Gênero','Contagem','Batch']))

    os.makedirs(output_path, exist_ok=True)
    resumo_path = os.path.join(output_path, 'resumo_familia_genero_por_batch.csv')
    resumo_df.to_csv(resumo_path, index=False)

    # Log amigável de problemas encontrados (se houver)
    if problemas:
        print("Atenção: problemas detectados em alguns batches:")
        for b, msg in problemas:
            print(f" - {b}: {msg}")

    return resumo_df