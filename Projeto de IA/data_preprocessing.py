
import pandas as pd
import numpy as np

# Caminho para os datasets
data_path = 'brazilian-ecommerce-ecommerce/'

# Carregar os datasets relevantes
customers = pd.read_csv(data_path + 'olist_customers_dataset.csv')
geolocation = pd.read_csv(data_path + 'olist_geolocation_dataset.csv')
order_items = pd.read_csv(data_path + 'olist_order_items_dataset.csv')
order_payments = pd.read_csv(data_path + 'olist_order_payments_dataset.csv')
order_reviews = pd.read_csv(data_path + 'olist_order_reviews_dataset.csv')
orders = pd.read_csv(data_path + 'olist_orders_dataset.csv')
products = pd.read_csv(data_path + 'olist_products_dataset.csv')
sellers = pd.read_csv(data_path + 'olist_sellers_dataset.csv')
category_translation = pd.read_csv(data_path + 'product_category_name_translation.csv')

print('Datasets carregados com sucesso!')

# --- Pré-processamento e Feature Engineering (baseado no projeto existente) ---

# 1. Juntar orders e order_reviews
orders_reviews = pd.merge(orders, order_reviews, on='order_id', how='left')

# 2. Converter colunas de data para datetime
date_cols = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'review_creation_date',
    'review_answer_timestamp'
]
for col in date_cols:
    orders_reviews[col] = pd.to_datetime(orders_reviews[col])

# 3. Calcular delivery_time_days e delay_days
orders_reviews['delivery_time_days'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_purchase_timestamp']).dt.days
orders_reviews['delay_days'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days

# 4. Classificar atrasos
def classify_delay(row):
    if pd.isna(row['delay_days']):
        return np.nan
    elif row['delay_days'] < -3:
        return 'Antecipado'
    elif row['delay_days'] >= -3 and row['delay_days'] <= 0:
        return 'No Prazo'
    elif row['delay_days'] > 0 and row['delay_days'] <= 7:
        return 'Atrasado'
    else:
        return 'Muito Atrasado'

orders_reviews['delay_category'] = orders_reviews.apply(classify_delay, axis=1)

# 5. Filtrar apenas pedidos entregues com review_score válido
processed_df = orders_reviews.dropna(subset=['review_score', 'order_delivered_customer_date']).copy()
processed_df['review_score'] = processed_df['review_score'].astype(int)

# 6. Definir o rótulo supervisionado de satisfação (review_ruim 1-3 vs review_boa 4-5)
processed_df['satisfaction_label'] = processed_df['review_score'].apply(lambda x: 'review_ruim' if x <= 3 else 'review_boa')

# 7. Juntar com order_items e products para obter informações de produto e categoria
# Primeiro, juntar order_items com products
order_items_products = pd.merge(order_items, products, on='product_id', how='left')

# Juntar com category_translation
order_items_products = pd.merge(order_items_products, category_translation, on='product_category_name', how='left')

# Agrupar por order_id para ter uma linha por pedido com informações agregadas dos itens
# Por simplicidade, vamos pegar a categoria mais comum ou a primeira categoria de produto por pedido
# Ou, para evitar complexidade, podemos apenas juntar o primeiro item de cada pedido
# Para este N1, vamos focar nas informações de atraso e review, como no projeto original.
# O projeto original não detalha o uso de product/item info para o N1, apenas para N2.
# Então, vamos manter o dataset focado em orders e reviews por enquanto.

# Salvar o dataset processado
processed_df.to_csv('dataset_processado_atrasos.csv', index=False)

print('\nDataset processado salvo em dataset_processado_atrasos.csv')

# --- Análise Exploratória Adicional (para o relatório) ---

# Estatísticas gerais
total_orders_analyzed = len(processed_df)
avg_delivery_time = processed_df['delivery_time_days'].mean()
avg_delay = processed_df['delay_days'].mean()
avg_review_score = processed_df['review_score'].mean()

print(f'\nTotal de pedidos analisados: {total_orders_analyzed}')
print(f'Tempo médio de entrega: {avg_delivery_time:.1f} dias')
print(f'Atraso médio (em relação à previsão): {avg_delay:.1f} dias')
print(f'Nota média de review: {avg_review_score:.2f}')

# Distribuição por categoria de atraso
delay_distribution = processed_df['delay_category'].value_counts(normalize=True) * 100
print('\nDistribuição por Categoria de Atraso:')
print(delay_distribution)

# Notas médias por categoria de atraso
avg_review_by_delay = processed_df.groupby('delay_category')['review_score'].mean().sort_values(ascending=False)
print('\nNota Média de Review por Categoria de Atraso:')
print(avg_review_by_delay)

# Correlação entre atraso e review_score
correlation = processed_df['delay_days'].corr(processed_df['review_score'])
print(f'\nCorrelação entre delay_days e review_score: {correlation:.3f}')

# Distribuição de notas por categoria de atraso (para replicar a tabela do projeto original)
review_distribution_by_delay = processed_df.groupby('delay_category')['review_score'].value_counts(normalize=True).unstack() * 100
print('\nDistribuição de Notas por Categoria de Atraso (%):')
print(review_distribution_by_delay)

# Salvar um resumo detalhado para o relatório
with open('eda_detailed_summary.txt', 'w') as f:
    f.write('Resumo Detalhado da Análise Exploratória e Pré-processamento:\n\n')
    f.write(f'Total de pedidos analisados: {total_orders_analyzed}\n')
    f.write(f'Tempo médio de entrega: {avg_delivery_time:.1f} dias\n')
    f.write(f'Atraso médio (em relação à previsão): {avg_delay:.1f} dias\n')
    f.write(f'Nota média de review: {avg_review_score:.2f}\n\n')
    f.write('Distribuição por Categoria de Atraso:\n')
    f.write(delay_distribution.to_string())
    f.write('\n\nNota Média de Review por Categoria de Atraso:\n')
    f.write(avg_review_by_delay.to_string())
    f.write(f'\n\nCorrelação entre delay_days e review_score: {correlation:.3f}\n\n')
    f.write('Distribuição de Notas por Categoria de Atraso (%):\n')
    f.write(review_distribution_by_delay.to_string())

print('\nResumo detalhado da EDA salvo em eda_detailed_summary.txt')


