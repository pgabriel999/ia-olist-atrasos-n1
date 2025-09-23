
import pandas as pd

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

# Exibir informações básicas de alguns datasets
print('\n--- orders.info() ---')
orders.info()
print('\n--- order_reviews.info() ---')
order_reviews.info()
print('\n--- customers.info() ---')
customers.info()

# Exibir as primeiras linhas de alguns datasets
print('\n--- orders.head() ---')
print(orders.head())
print('\n--- order_reviews.head() ---')
print(order_reviews.head())
print('\n--- customers.head() ---')
print(customers.head())

# Verificar valores ausentes
print('\n--- Valores ausentes em orders ---')
print(orders.isnull().sum())
print('\n--- Valores ausentes em order_reviews ---')
print(order_reviews.isnull().sum())

# Merge inicial para começar a replicar a análise do projeto existente
# Juntar orders e order_reviews
orders_reviews = pd.merge(orders, order_reviews, on='order_id', how='left')

# Juntar com customers para ter informações do cliente
orders_reviews_customers = pd.merge(orders_reviews, customers, on='customer_id', how='left')

print('\n--- orders_reviews_customers.info() após merges ---')
orders_reviews_customers.info()

print('\n--- orders_reviews_customers.head() após merges ---')
print(orders_reviews_customers.head())

# Salvar um pequeno resumo para o relatório
with open('eda_summary.txt', 'w') as f:
    f.write('Resumo da Análise Exploratória Inicial:\n\n')
    f.write('Informações sobre o dataset orders:\n')
    orders.info(buf=f)
    f.write('\n\nInformações sobre o dataset order_reviews:\n')
    order_reviews.info(buf=f)
    f.write('\n\nInformações sobre o dataset customers:\n')
    customers.info(buf=f)
    f.write('\n\nValores ausentes em orders:\n')
    f.write(orders.isnull().sum().to_string())
    f.write('\n\nValores ausentes em order_reviews:\n')
    f.write(order_reviews.isnull().sum().to_string())
    f.write('\n\norders_reviews_customers.info() após merges:\n')
    orders_reviews_customers.info(buf=f)

print('\nResumo da EDA inicial salvo em eda_summary.txt')


