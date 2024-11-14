import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Load the data
df = pd.read_csv('medical_examination.csv')

# 2: Add an overweight column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3: Normalize cholesterol and gluc data (make 0 good and 1 bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4: Define the draw_cat_plot function
def draw_cat_plot():
    # 5: Create DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6: Group and reformat data to split it by cardio and count values
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7: Create a seaborn catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 9: Save figure
    fig.savefig('catplot.png')
    return fig


# 10: Define the draw_heat_map function
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
        ]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15: Draw the heatmap with seaborn
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                cbar_kws={'shrink': .5}, ax=ax)

    # 16: Save figure
    fig.savefig('heatmap.png')
    return fig
