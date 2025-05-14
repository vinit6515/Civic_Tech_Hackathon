# Sample chatbot data loading
import pandas as pd

df = pd.read_csv('indianapolis_issues.csv')
knowledge_base = {
    row['Category']: {
        'value': row['Value'],
        'description': row['Description'],
        'source': row['Source URL']
    } for _, row in df.iterrows()
}

