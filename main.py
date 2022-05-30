import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import class_leveling
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as CM
from sklearn.neural_network import MLPRegressor as MLPR


# mpl.use('agg')


def main(base_dir):
    df, class_df = get_df(base_dir)
    # class_leveling.show_classification_balance(class_df)
    # class_leveling.show_balance_on_two_type(class_df, ['Aligned', 'Flocking'])
    # class_leveling.show_balance_on_type(class_df, 'Flocking')
    # class_leveling.show_balance_on_type(class_df, 'Grouped')
    # show_correlation(df, class_df)
    # pred_vel(df, class_df)

    for group_num in range(15287, len(df.index)):
        show_movement(group_num, df, class_df)
        # show_scatter(group_num, df, class_df, s=3, color='red')
        pass


def pred(df, class_df):
    class_df = class_df.apply(lambda x: str(tuple(x)), axis=1).map(
        {key: val for val, key in enumerate(set(class_df.apply(
            lambda x: str(tuple(x)), axis=1)))})
    # class_df = class_df['Flocking']
    X_train, X_test, Y_train, Y_test = tts(df, class_df, test_size=0.3)
    model = RFC(n_estimators=50)
    model.fit(X_train, Y_train)
    labels = model.predict(X_test)
    print(CM(labels, Y_test))
    pass


def pred_vel(df, class_df):
    # df = pd.concat([df, class_df], axis=1)
    df = df[list(class_df.apply(lambda x: x.all(), axis=1))]
    # df = df[list(class_df.apply(lambda x: (x[0] == 0) and (x[1] == 1) and (x[2] == 0), axis=1))]
    tmp = [[f'xVel{i}', f'yVel{i}'] for i in range(1, 201)]
    class_list = []
    for i_list in tmp:
        class_list.extend(i_list)
    Y = df[class_list].copy()
    X = df.drop(class_list, axis=1)
    for col_name in X.columns:
        X[col_name] /= X[col_name].mean()
    '''
    row_even = list(
        pd.Series(range(400)).apply(lambda x: x % 2 == 0))
    row_odd = list(
        pd.Series(range(400)).apply(lambda x: x % 2 == 1))
    for i, row in Y.iterrows():
        # + min is mammking weird things
        Y_even = row[row_even] / (row[row_even] - row[row_even].min()).median() # + row[row_even].min()
        Y_odd = row[row_odd] / (row[row_odd] - row[row_odd].min()).median() # + row[row_odd].min()
        # Y_even = row[row_even] / row[row_even].quantile(0.99)
        # Y_odd = row[row_odd] / row[row_odd].quantile(0.99)
        Y.loc[i, row_even] = Y_even
        Y.loc[i, row_odd] = Y_odd
    '''

    X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.3)
    model = MLPR(hidden_layer_sizes=(200, 200), alpha=0.001)
    print('starting...')
    model.fit(X_train, Y_train)
    print(model.score(X_test, Y_test))
    print(model.score(X_train, Y_train))
    print(((model.predict(X_test) - Y_test).applymap(abs) > 0.5).sum().sum() / (Y_test.shape[1] * Y_test.shape[0]))
    pass


def show_correlation(df: pd.DataFrame, class_df):
    row_len = df.shape[0] * 200
    col_len = df.shape[1] / 200
    tmp_df = pd.DataFrame(df.values.reshape(int(row_len), int(col_len)))
    tmp_ts = tmp_df.iloc[:, 2:4].apply(
        lambda x: np.linalg.norm(x.values),
        axis=1)
    class_df = class_df.apply(tuple, axis=1)
    class_df = class_df.iloc[np.arange(len(class_df)).repeat(200)]
    class_df.index = pd.Index([i for i in range(len(class_df))])
    tmp_ts.index = pd.Index([i for i in range(len(tmp_ts))])
    tmp_ts['class'] = class_df
    new_gb = tmp_ts.groupby('class')
    i = 0
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(nrows=1, ncols=len(new_gb.groups.keys()))
    ax = fig.add_subplot(gs[0, i])
    for name, small_df in new_gb:
        ax = fig.add_subplot(gs[0, i], sharey=ax)
        ax.violinplot(list(small_df))
        ax.set_title(name)
        i += 1
    plt.show()


def get_df(base_dir):
    res = []
    class_res = []
    for class_type in base_dir:
        name_class, csv_file = class_type
        tmp = pd.read_csv(csv_file, na_values=' ')
        if tmp.isna().any().any():
            tmp.iloc[-1, 0] = 831.14
            tmp1 = tmp.loc[17293:].copy()
            tmp1.index = tmp1.index + 1
            tmp1.loc[17293] = tmp1.iloc[-1]
            tmp1.drop(max(tmp1.index), inplace=True)
            tmp1.sort_index(inplace=True)
            tmp = pd.concat([tmp[:17293], tmp1])
        df, ts = tmp.iloc[:, :-1], tmp.iloc[:, -1]
        ts.name = name_class
        res.append(df)
        class_res.append(ts)
    # for i in range(1, len(res)):
    #    if not res[i].equals(res[i-1]):
    #        raise ValueError('different dataset for same classification')
    class_df = pd.concat(class_res, axis=1)
    class_df = class_df[['Aligned', 'Flocking', 'Grouped']]
    res = res[0]
    return res, class_df


def get_one_scene(group_num, df, classification_df):
    res = df.loc[group_num]
    classification = classification_df.loc[group_num]
    res = pd.DataFrame(res.values.reshape(200, 12))
    # sort the boids according to their angle
    res['theta'] = res[[0, 1]].apply(lambda x: np.arctan2(x[1], x[0]), axis=1)
    res.sort_values(by='theta', inplace=True)
    res.drop('theta', axis=1, inplace=True)
    # making color for bar plots
    rad_cmap = plt.get_cmap('hsv')
    color_list = pd.Series(np.linspace(0, 1, 200)).apply(rad_cmap)
    return res, classification, color_list


def show_movement(group_num, df, classification_df):
    res, classification, color_ts = get_one_scene(
        group_num, df, classification_df)
    # getting only location and velocity
    res = res.iloc[:, :4]
    frames_num = 200
    res, anim_name = move_fract(res, classification, time_frame=frames_num, fract=0.00003)
    color_discrete_map = {i: f'rgb({c[0]}, {c[1]}, {c[2]})'
                          for i, c in color_ts.iteritems()}
    color_ts = pd.concat([
        pd.Series([i for i in color_ts.index])
        for _ in range(frames_num)])
    fig = px.scatter(
        res,
        x=0, y=1,
        animation_frame=anim_name,
        color_discrete_map=color_discrete_map,
        color=color_ts,
    )
    fig.layout.updatemenus[0].buttons[0]. \
        args[1]["frame"]["duration"] = 1
    fig.layout.updatemenus[0].buttons[0]. \
        args[1]['transition']['duration'] = 1
    # fig.show()
    fig.write_html(r"C:\Users\yair\Desktop\data_science\figs\movement\num_" + f'{group_num}.html')


# small movement toward velocity
def move_fract(df: pd.DataFrame, classification_df: pd.DataFrame, time_frame=100, fract=0.00005):
    fract = fract * df.max().max()
    vel_df = df.iloc[:, 2:] * fract
    df = df.iloc[:, :2].copy()
    animation_name = 'animation'
    df[animation_name] = 0
    res = [df]
    tmp = df
    for i in range(1, time_frame):
        tmp = pd.DataFrame(tmp.iloc[:, :2].values + vel_df.values)
        tmp[animation_name] = i
        res.append(tmp)
    res = pd.concat(res, axis=0)
    return res, animation_name


def show_scatter(group_num, df, class_df, s=4, color='green'):
    res, classification, color_ts = get_one_scene(
        group_num, df, class_df)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(nrows=3, ncols=6, height_ratios=[1.5, 1, 1])
    for num, name in enumerate(['Point', 'Velocity']):
        ax_place = fig.add_subplot(gs[0, num * 3: num * 3 + 3])
        ax_place.scatter(res.iloc[:, num * 2], res.iloc[:, num * 2 + 1], s=s, c=color_ts)
        ax_place.set_title(name)
        for spine_name in ['left', 'right', 'top', 'bottom']:
            ax_place.spines[spine_name].set_position('zero')
        ax_place.patch.set_edgecolor('black')
        ax_place.patch.set_linewidth('1')
    for num, name in enumerate(['Alignment', 'Separation', 'Coherence']):
        ax_place = fig.add_subplot(gs[1, num * 2: num * 2 + 2])
        num += 2
        ax_place.scatter(res.iloc[:, num * 2], res.iloc[:, num * 2 + 1], s=s, c=color_ts)
        ax_place.set_title(name)
        for spine_name in ['left', 'right', 'top', 'bottom']:
            ax_place.spines[spine_name].set_position('zero')
        ax_place.patch.set_edgecolor('black')
        ax_place.patch.set_linewidth('1')
    # adding bar plots with shared y
    ax_place = fig.add_subplot(gs[2, 0: 3])
    for i in range(200):
        ax_place.bar(i, res.iloc[i, 10], color=color_ts.iloc[i])
    ax_place.set_title('N_Alignment')
    ax_place.tick_params(axis='y', which='minor', left=False)
    ax_place = fig.add_subplot(gs[2, 3: 6], sharey=ax_place)
    ax_place.set_yticks([i * 2 for i in range(100)], minor=True)
    plt.setp(ax_place.get_yticklabels(), visible=False)
    for i in range(200):
        ax_place.bar(i, res.iloc[i, 11], color=color_ts.iloc[i])
    ax_place.set_title('N_Separation')
    scenario_tilte = f'id:{group_num}'
    fig.suptitle(scenario_tilte, color='darkgreen')
    class_types_list = ['Aligned', 'Flocking', 'Grouped']
    for i in range(3):
        classification_stat = classification[i] == 1
        fig.text(
            0.404 + i * 0.095, 0.93,
            f'{class_types_list[i]}={classification_stat}',
            ha='center', va='bottom',
            color='blue' if classification_stat else 'red',
            size='large',
        )
    print(scenario_tilte)
    plt.savefig(r"C:\Users\yair\Desktop\data_science\figs\pic\num_" + f'{group_num}')
    plt.close(fig)


if __name__ == '__main__':
    base_dir = [
        ('Aligned', r"C:\Users\yair\Desktop\data_science\Swarm Behavior Data\Aligned.csv"),
        ('Flocking', r"C:\Users\yair\Desktop\data_science\Swarm Behavior Data\Flocking.csv"),
        ('Grouped', r"C:\Users\yair\Desktop\data_science\Swarm Behavior Data\Grouped.csv"),
    ]
    main(base_dir)
