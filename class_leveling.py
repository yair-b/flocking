import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


def show_classification_porportion(class_df):
    class_type = ['Alinged', 'Flocking', 'grouped']
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(nrows=3, ncols=3)
    ax = fig.add_subplot(gs[0, 0: 3])
    df_all = \
        pd.Series(
            0,
            index=[(i, j, k) for i in range(2)
                   for j in range(2) for k in range(2)],
        )
    df_all = df_all.add(
        class_df. \
            apply(lambda x: tuple(x), axis=1). \
            value_counts(),
        fill_value=0
    ).sort_index()
    ax.bar([s for s in map(str, df_all.index)], df_all + 0.02, color='red')
    ax.set_title(f'({class_type[0]} {class_type[1]} {class_type[2]})')

    for i in range(2):
        class_df_tmp = class_df.copy()
        class_df_tmp.columns = [0, 1, 2]
        first_df = class_df_tmp[i]
        sec_df = class_df_tmp.loc[:,i+1:].columns
        for l in sec_df:
            ax = fig.add_subplot(gs[1, i+l-1])
            df_all = \
                pd.Series(
                    0,
                    index=[(j, k) for j in range(2) for k in range(2)],
                )
            df_all = df_all.add(
                class_df_tmp[[i, l]]. \
                    apply(lambda x: tuple(x), axis=1). \
                    value_counts(),
                fill_value=0
            ).sort_index()
            ax.bar([s for s in map(str, df_all.index)], df_all + 0.03, color='blue')
            ax.set_title(f'({class_type[i]} {class_type[l]})')

    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        df_all = \
            pd.Series(
                0,
                index=[j for j in range(2)],
            )
        df_all = df_all.add(
            class_df_tmp.iloc[:, i]. \
                value_counts(),
            fill_value=0
        ).sort_index()
        ax.bar([s for s in map(str, df_all.index)], df_all + 0.02, color='green')
        ax.set_title(f'{class_type[i]}')
    plt.subplots_adjust(hspace=1)
    plt.savefig(r"C:\Users\yair\Desktop\data_science\figs\classification_porp", ppi=100)
    plt.show()

if __name__ == "__main__":
    df = pd.DataFrame({'a': [0, 1, 0, 0, 0, 1, 1, 1], 'b': [0, 0, 0, 0, 1, 1, 1, 0],
                       'c': [1, 1, 1, 1, 1, 0, 1, 0]})
    show_classification_porportion(class_df=df)
