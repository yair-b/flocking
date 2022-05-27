import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import plotly.express as px
import class_leveling


# mpl.use('agg')


def main(base_dir):
    df, class_df = get_df(base_dir)
    class_levelilng.show_classification_porportion(class_df)
  #  for group_num in range(6000, len(df.index)):
        #show_movement(group_num, df, class_df)
  #      show_scatter(group_num, df, class_df, s=3, color='red')


def get_df(base_dir):
    res = []
    class_res = []
    for class_type in base_dir:
        name_class, csv_file = class_type
        tmp = pd.read_csv(csv_file, na_values=' ')
        df, ts = tmp.iloc[:, :-1], tmp.iloc[:, -1]
        ts.name = name_class
        res.append(df)
        class_res.append(ts)
    # for i in range(1, len(res)):
    #    if not res[i].equals(res[i-1]):
    #        raise ValueError('different dataset for same classification')
    class_df = pd.concat(class_res, axis=1)
    res = res[0]
    return res, class_df


def get_one_scene(group_num, df, classification_df):
    res = df.iloc[group_num]
    classification = classification_df.iloc[group_num]
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
    class_types_list = ['Aligned', 'Flocking', 'grouped']
    for i in range(3):
        classification_stat = classification[i] == 1
        fig.text(
            0.404 + i*0.095, 0.93,
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
