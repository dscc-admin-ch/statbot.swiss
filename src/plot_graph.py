import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import pandas as pd
import numpy as np


def plot_by_db_id(
    filepath,
    save_path=None,
    show=False,
    title=None,
    en_color="cyan",
    de_color="#44bd32",
):
    """
    Plot the graph of a single db_id
    :param db_id: the db_id of the graph
    :param save_path: the path to save the graph
    :param show: whether to show the graph
    :return:
    """
    # Load the data
    if isinstance(filepath, str):
        filepath = Path(filepath)
    data = pd.read_csv(filepath)
    db_lang_dict = data[["db_id", "lang"]].set_index("db_id").to_dict()["lang"]
    if "max_score" in data.columns:
        db_max_mean_similarity_df = 1 - data.groupby("db_id")["max_score"].mean()
    if "avg_score" in data.columns:
        db_avg_mean_similarity_df = 1 - data.groupby("db_id")["avg_score"].mean()
    _filter = data["label"] == True
    db_id_df = (
        data[_filter].groupby(data["db_id"]).count()
        / data.groupby(data["db_id"]).count()
    )[["label"]].fillna(0)
    db_id_df = db_id_df.rename(columns={"label": "acc"})
    db_id_df["lang"] = db_id_df.index.map(db_lang_dict)
    if "avg_score" in data.columns:
        db_id_df["mean_avg_similarity"] = db_avg_mean_similarity_df
    if "max_score" in data.columns:
        db_id_df["mean_max_similarity"] = db_max_mean_similarity_df

    sorted_db_id_df = db_id_df.sort_values(["lang", "acc"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = {"en": en_color, "de": de_color}
    sorted_db_id_df.reset_index().plot.bar(
        "db_id", "acc", color=sorted_db_id_df["lang"].replace(colors), ax=ax
    ).legend(
        [Patch(facecolor=colors["en"]), Patch(facecolor=colors["de"])],
        ["acc(en)", "acc(de)"],
    )
    """
    sorted_db_id_df.reset_index().plot(
        x="db_id", y="mean_max_similarity", ax=ax, secondary_y=True, color="red", rot=90
    )
    sorted_db_id_df.reset_index().plot(
        x="db_id",
        y="mean_avg_similarity",
        linestyle="--",
        dashes=(5, 5),
        ax=ax,
        secondary_y=True,
        color="blue",
        rot=90,
    )
    """
    ax.set_ylabel("Accuracy")
    # ax.right_ax.set_ylabel("Example Similarity")
    if not title:
        title = "dev samples"
    ax.title.set_text(f"Accuracy and Example Similarity of {title}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_by_hardness(
    filepath,
    save_path=None,
    show=False,
    title=None,
    en_color="cyan",
    de_color="#44bd32",
):
    # Load the data
    if isinstance(filepath, str):
        filepath = Path(filepath)
    data = pd.read_csv(filepath)
    db_lang_dict = data[["db_id", "lang"]].set_index("db_id").to_dict()["lang"]
    if "max_score" in data.columns:
        db_max_mean_similarity_df = (
            1 - data.groupby(["hardness", "lang"])["max_score"].mean()
        )
    if "avg_score" in data.columns:
        db_avg_mean_similarity_df = (
            1 - data.groupby(["hardness", "lang"])["avg_score"].mean()
        )
    _filter = data["label"] == True
    db_id_df = (
        data[_filter].groupby(["hardness", "lang"]).count()
        / data.groupby(["hardness", "lang"]).count()
    )[["label"]].fillna(0)
    db_id_df = db_id_df.rename(columns={"label": "acc"})
    db_id_df["lang"] = db_id_df.index.map(db_lang_dict)
    if "max_score" in data.columns:
        db_id_df["mean_avg_similarity"] = db_avg_mean_similarity_df
    if "avg_score" in data.columns:
        db_id_df["mean_max_similarity"] = db_max_mean_similarity_df
    db_id_df.drop(columns=["lang"], inplace=True)
    sorted_db_id_df = db_id_df.sort_values(["lang", "acc"], ascending=[False, False])
    sorted_db_id_df.reset_index(inplace=True)
    sorted_db_id_df["hardness-lang"] = (
        sorted_db_id_df["hardness"] + "-" + sorted_db_id_df["lang"]
    )
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = {"en": en_color, "de": de_color}
    sorted_db_id_df.reset_index().plot.bar(
        "hardness-lang", "acc", color=sorted_db_id_df["lang"].replace(colors), ax=ax
    ).legend(
        [Patch(facecolor=colors["en"]), Patch(facecolor=colors["de"])],
        ["acc(en)", "acc(de)"],
    )
    """
    sorted_db_id_df.reset_index().plot(
        x="hardness-lang",
        y="mean_max_similarity",
        ax=ax,
        secondary_y=True,
        color="red",
        rot=45,
    )
    sorted_db_id_df.reset_index().plot(
        x="hardness-lang",
        y="mean_avg_similarity",
        linestyle="--",
        dashes=(5, 5),
        ax=ax,
        secondary_y=True,
        color="blue",
        rot=45,
    )
    """
    ax.set_ylabel("Accuracy")
    # ax.right_ax.set_ylabel("Example Similarity")
    if not title:
        title = "dev samples"
    ax.title.set_text(f"Accuracy of {title}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_by_lang(
    filepath,
    save_path=None,
    show=False,
    title=None,
    en_color="cyan",
    de_color="#44bd32",
):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    data = pd.read_csv(filepath)
    _filter = data["label"] == True
    lang_df = (
        (
            data[_filter].groupby(data["lang"]).count()
            / data.groupby(data["lang"]).count()
        )[["label"]]
    ).fillna(0)
    lang_df = lang_df.rename(columns={"label": "acc"})
    sorted_lang_df = lang_df.sort_values(["lang", "acc"], ascending=[False, False])
    sorted_lang_df.reset_index(inplace=True)

    fig, ax = plt.subplots(figsize=(20, 10))
    colors = {"en": en_color, "de": de_color}
    sorted_lang_df.reset_index().plot.bar(
        "lang", "acc", color=sorted_lang_df["lang"].replace(colors), ax=ax
    ).legend(
        [Patch(facecolor=colors["en"]), Patch(facecolor=colors["de"])],
        ["acc(en)", "acc(de)"],
    )
    ax.set_ylabel("Accuracy")
    # ax.right_ax.set_ylabel("Example Similarity")
    if not title:
        title = "dev samples"
    ax.title.set_text(f"Accuracy of {title}")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0, ha="right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show() if show else plt.close()