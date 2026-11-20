import json
from itertools import repeat
from os import PathLike
from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pytorch_fob.engine.parser import YAMLParser
from pytorch_fob.engine.utils import (
    AttributeDict,
    convert_type_inside_dict,
    log_debug,
    log_info,
    log_warn,
)
from pytorch_fob.evaluation import evaluation_path


def get_available_trials(dirname: Path, config: AttributeDict, depth: int = 1):
    """finds the path for all trials in the *dirname* directory"""
    # RECURSIVELY FIND ALL DIRS IN DIRNAME (up to depth)
    assert isinstance(dirname, Path)
    subdirs: list[Path] = [dirname]
    all_results_must_be_same_depth = True
    for _ in range(depth):
        if all_results_must_be_same_depth:
            new_subdirs: list[Path] = []
            for subdir in subdirs:
                new_subdirs += [x for x in subdir.iterdir() if x.is_dir()]
            subdirs = new_subdirs
        else:
            for subdir in subdirs:
                subdirs += [x for x in subdir.iterdir() if x.is_dir()]
    format_str = "\n  "  # f-string expression part cannot include a backslash
    log_debug(
        f"found the following directories:{format_str}{format_str.join(str(i) for i in subdirs)}."
    )

    def is_trial(path: Path):
        # here we could do additional checks to filter the subdirectories
        # currently we only check if there is a config file
        for x in path.iterdir():
            found_a_config_file = x.name == config.experiment_files.config
            if found_a_config_file:
                return True
        return False

    subdirs = list(filter(is_trial, subdirs[::-1]))
    log_debug(
        f"We assume the following to be trials:{format_str}{format_str.join(str(i) for i in subdirs)}."
    )
    return subdirs


def dataframe_from_trials(
    trial_dir_paths: List[Path], config: AttributeDict
) -> pd.DataFrame:
    """takes result from get_available_trials and packs them in a dataframe,
    does not filter duplicate hyperparameter settings."""
    dfs: List[pd.DataFrame] = []

    for path in trial_dir_paths:

        config_file = path / config.experiment_files.config
        if config.last_instead_of_best:
            result_file = path / config.experiment_files.last_model
        else:
            result_file = path / config.experiment_files.best_model
        all_files_exist = all([config_file.is_file(), result_file.is_file()])
        if not all_files_exist:
            log_warn(
                f"WARNING: one or more files are missing in {path}. Skipping this hyperparameter setting."
                + f"  <{config_file}>: {config_file.is_file()} and\n  <{result_file}>: {result_file.is_file()})"
            )
            continue

        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(config_file)
        # convert the sub dicts first, then the dict itself
        yaml_content = convert_type_inside_dict(
            yaml_content, src=dict, tgt=AttributeDict
        )
        yaml_content = AttributeDict(yaml_content)

        # use user given value
        metric_of_value_to_plot = config.plot.metric

        # compute it if user has not given a value
        if not metric_of_value_to_plot:
            raise ValueError("evaluation.plot.metric is not set")

        data = pd.json_normalize(yaml_content)

        with open(result_file, "r", encoding="utf8") as f:
            content = json.load(f)
            if metric_of_value_to_plot in content[0]:
                data.at[0, metric_of_value_to_plot] = content[0][
                    metric_of_value_to_plot
                ]
            else:
                log_warn(f"could not find value for {metric_of_value_to_plot} in json")

        dfs.append(data)

    if len(dfs) == 0:
        raise ValueError("no dataframes found, check your config")
    df = pd.concat(dfs, sort=False)

    return df


def create_matrix_plot(
    dataframe: pd.DataFrame,
    config: AttributeDict,
    cols: str,
    idx: str,
    ax=None,
    cbar: bool = True,
    vmin: None | int = None,
    vmax: None | int = None,
):
    """
    Creates one heatmap and puts it into the grid of subplots.
    Uses pd.pivot_table() and sns.heatmap().
    """
    df_entry = dataframe.iloc[0]
    metric_name = df_entry["evaluation.plot.metric"]

    # CLEANING LAZY USER INPUT
    # cols are x-axis, idx are y-axis
    if cols not in dataframe.columns:
        log_warn(
            "x-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n"
            + f"  using '{'optimizer.' + cols}' as 'x-axis' instead."
        )
        cols = "optimizer." + cols
    if idx not in dataframe.columns:
        log_warn(
            "y-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n"
            + f"  using '{'optimizer.' + idx}' as 'y-axis' instead."
        )
        idx = "optimizer." + idx
    # create pivot table and format the score result
    pivot_table = pd.pivot_table(
        dataframe, columns=cols, index=idx, values=metric_name, aggfunc="mean"
    )

    fmt = None
    format_string = dataframe["evaluation.plot.format"].iloc[0]

    # scaline the values given by the user to fit his format needs (-> and adapting the limits)
    value_exp_factor, decimal_points = format_string.split(".")
    value_exp_factor = int(value_exp_factor)
    decimal_points = int(decimal_points)
    if vmin:
        vmin *= 10**value_exp_factor
    if vmax:
        vmax *= 10**value_exp_factor
    pivot_table = (pivot_table * (10**value_exp_factor)).round(decimal_points)
    fmt = f".{decimal_points}f"

    # up to here limits was the min and max over all dataframes,
    # usually we want to use user values
    if "evaluation.plot.limits" in dataframe.columns:
        limits = dataframe["evaluation.plot.limits"].iloc[0]
        if limits:
            vmin = min(limits)
            vmax = max(limits)
            log_debug(f"setting cbar limits to {vmin}, {vmax} ")

    colormap_name = config.plotstyle.color_palette
    low_is_better = dataframe["evaluation.plot.test_metric_mode"].iloc[0] == "min"
    if low_is_better:
        colormap_name += "_r"  # this will "inver" / "flip" the colorbar
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    metric_legend = pretty_name(metric_name)

    # FINETUNE POSITION
    # left bottom width height
    # cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])
    cbar_ax = None

    if not config.plot.std:
        return sns.heatmap(
            pivot_table,
            ax=ax,
            cbar_ax=cbar_ax,
            annot=True,
            fmt=fmt,
            annot_kws={"fontsize": config.plotstyle.matrix_font.size},
            cbar=cbar,
            vmin=vmin,
            vmax=vmax,
            cmap=colormap,
            cbar_kws={"label": f"{metric_legend}"},
        )
    else:
        # BUILD STD TABLE
        pivot_table_std = pd.pivot_table(
            dataframe,
            columns=cols,
            index=idx,
            values=metric_name,
            aggfunc=config.plot.aggfunc,
            fill_value=float("inf"),
            dropna=False,
        )
        if float("inf") in pivot_table_std.values.flatten():
            log_warn(
                "WARNING: Not enough data to calculate the std, skipping std in plot"
            )

        pivot_table_std = (pivot_table_std * (10**value_exp_factor)).round(
            decimal_points
        )

        annot_matrix = pivot_table.copy().astype("string")
        for i in pivot_table.index:
            for j in pivot_table.columns:
                mean = pivot_table.loc[i, j]
                std = pivot_table_std.loc[i, j]
                std_string = f"\n±({round(std, decimal_points)})" if std != float("inf") else ""  # type: ignore
                annot_matrix.loc[i, j] = f"{round(mean, decimal_points)}{std_string}"  # type: ignore

        fmt = ""  # cannot format like before, as we do not only have a number

        return sns.heatmap(
            pivot_table,
            ax=ax,
            cbar_ax=cbar_ax,
            annot=annot_matrix,
            fmt=fmt,
            annot_kws={"fontsize": config.plotstyle.matrix_font.size},
            cbar=cbar,
            vmin=vmin,
            vmax=vmax,
            cmap=colormap,
            cbar_kws={"label": f"{metric_legend}"},
        )


def get_all_num_rows_and_their_names(dataframe_list: list[pd.DataFrame], config):
    n_rows: list[int] = []
    row_names: list[list[str]] = []
    for i, df in enumerate(dataframe_list):
        x_axis = config.plot.x_axis[i]
        y_axis = config.plot.y_axis[0]
        metrics = df["evaluation.plot.metric"].unique()
        ignored_cols = [x_axis, y_axis]
        ignored_cols += list(metrics)
        ignored_cols += config.get("ignore_keys", [])
        ignored_cols += config.get("aggregate_groups", [])
        current_n_rows, current_names = get_num_rows(df, ignored_cols, config)
        n_rows.append(current_n_rows)
        if not current_names:  # will be empty if we have only one row
            current_names.append("default")
        row_names.append(current_names)

    return n_rows, row_names


def get_num_rows(
    dataframe: pd.DataFrame, ignored_cols: list[str], config: AttributeDict
) -> tuple[int, list[str]]:
    """each matrix has 2 params (on for x and y each), one value, and we aggregate over seeds;
    if there are more than than these 4 parameter with different values,
    we want to put that in seperate rows instead of aggregating over them.
    returning: the number of rows (atleast 1) and the names of the cols"""
    necesarry_rows = 0

    # the user might specify a value for the groups that we should split on in <split_groups>
    whitelisted_cols: list[str] | Literal["all"] = (
        "all"  # everything is whitelisted if this value stays 'all'
    )
    if isinstance(config.split_groups, list):
        whitelisted_cols = config.split_groups[:]
    elif config.split_groups is False:
        whitelisted_cols = []

    columns_with_non_unique_values = []
    for col in dataframe.columns:
        is_eval_key = col.startswith("evaluation.")
        is_ignored = col in ignored_cols
        is_whitelisted = whitelisted_cols == "all" or col in whitelisted_cols
        if any([is_ignored, is_eval_key, not is_whitelisted]):
            if is_whitelisted:
                log_warn(
                    f"{col} is in the whitelist, but will be ignored. "
                    f"Probably {col} is in both 'split_groups' and 'aggregate_groups'."
                )
            log_debug(f"ignoring {col}")
            continue
        nunique = dataframe[col].nunique(dropna=False)
        if nunique > 1:
            log_debug(f"adding {col} since there are {nunique} unique values")
            for unique_hp in dataframe[col].unique():
                columns_with_non_unique_values.append(f"{col}={unique_hp}")
            necesarry_rows += (
                nunique  # each unique parameter should be an individual plot
            )

    rows_number = max(necesarry_rows, 1)
    col_names = columns_with_non_unique_values
    log_debug(f"{rows_number=}")
    log_debug(f"{col_names=}")

    return rows_number, col_names


def find_global_vmin_vmax(dataframe_list, config):
    vmin: int | float | None = None
    vmax: int | float | None = None
    num_cols = len(dataframe_list)

    if num_cols > 1:
        # all subplots should have same colors -> we need to find the limits
        vmin = float("inf")
        vmax = float("-inf")

        for i in range(num_cols):
            dataframe = dataframe_list[i]
            cols = config.plot.x_axis[i]
            idx = config.plot.y_axis[0]
            key = config.plot.metric

            pivot_table = pd.pivot_table(
                dataframe, columns=cols, index=idx, values=key, aggfunc="mean"
            )

            min_value_present_in_current_df = pivot_table.min().min()
            max_value_present_in_current_df = pivot_table.max().max()

            log_debug(
                "colorbar_limits:\n"
                + f"  subfigure number {i+1}, checking for metric {key}: \n"
                + f"  min value is {min_value_present_in_current_df},\n"
                + f"  max value is {max_value_present_in_current_df}"
            )
            vmin = min(vmin, min_value_present_in_current_df)
            vmax = max(vmax, max_value_present_in_current_df)

    return vmin, vmax


def create_figure(dataframe_list: list[pd.DataFrame], config: AttributeDict):
    """
    Takes a list of dataframes. Each dataframe is processed into a column of heatmaps.
    """
    num_cols: int = len(dataframe_list)

    # calculate the number of rows for each dataframe
    n_rows, row_names = get_all_num_rows_and_their_names(dataframe_list, config)

    # Handling of the number of rows in the plot
    # we could either create a full rectangular grid, or allow each subplot to nest subplots
    # for nesting we would need to create subfigures instead of subplots i think
    if config.split_groups is False:
        n_rows_max = 1
        row_names = [["default"] for _ in range(num_cols)]
    else:
        n_rows_max = max(n_rows)

    log_debug(f"{n_rows=} and {num_cols=}")

    # TODO, figsize was just hardcoded for (1, 2) grid and left to default for (1, 1) grid
    #       probably not worth the hazzle to create something dynamic (atleast not now)
    # EDIT: it was slightly adapted to allow num rows without being completely unreadable
    # margin = (num_subfigures - 1) * 0.3
    # figsize=(5*n_cols + margin, 2.5)
    scale = config.plotstyle.scale
    if num_cols == 1 and n_rows_max > 1:
        figsize = (2**3 * scale, 2 * 3 * n_rows_max * scale)
    elif num_cols == 2:
        # TODO: after removing cbar from left subifgure, it is squished
        #       there is an argument to share the legend, we should use that
        figsize = (12 * scale, 5.4 * n_rows_max * scale)
    elif num_cols > 2:
        figsize = (12 * (num_cols / 2) * scale, 5.4 * n_rows_max * scale)
    else:
        figsize = None

    # TODO: use seaborn FacetGrid
    fig, axs = plt.subplots(n_rows_max, num_cols, figsize=figsize)
    if n_rows_max == 1:
        axs = [axs]
    if num_cols == 1:
        axs = [[ax] for ax in axs]  # adapt for special case so we have unified types

    # Adjust left and right margins as needed
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.97, hspace=0.38, bottom=0.05,wspace=0.3)

    # None -> plt will chose vmin and vmax
    vmin, vmax = find_global_vmin_vmax(dataframe_list, config)

    for i in range(num_cols):
        num_nested_subfigures: int = n_rows[i]

        if not config.split_groups:
            create_one_grid_element(
                dataframe_list,
                config,
                axs,
                i,
                j=0,
                max_i=num_cols,
                max_j=0,
                vmin=vmin,
                vmax=vmax,
                n_rows=n_rows,
                row_names=row_names,
            )
        else:
            for j in range(num_nested_subfigures):
                create_one_grid_element(
                    dataframe_list,
                    config,
                    axs,
                    i,
                    j,
                    max_i=num_cols,
                    max_j=num_nested_subfigures,
                    vmin=vmin,
                    vmax=vmax,
                    n_rows=n_rows,
                    row_names=row_names,
                )

    if config.plotstyle.tight_layout:
        fig.tight_layout()
    # SUPTITLE (the super title on top of the whole figure in the middle)
    # # TODO super title might be squished when used together with tight layout (removing for now)
    # if n_rows_max > 1 or num_cols > 1:
    #     # set experiment name as title when multiple matrices in image
    #     if config.experiment_name:
    #         fig.suptitle(config.experiment_name)
    return fig, axs


def create_one_grid_element(
    dataframe_list: list[pd.DataFrame],
    config: AttributeDict,
    axs,
    i: int,
    j: int,
    max_i: int,
    max_j: int,
    vmin,
    vmax,
    n_rows,
    row_names,
):
    """does one 'axs' element as it is called in plt"""
    num_nested_subfigures: int = n_rows[i]
    name_for_additional_subplots: list[str] = row_names[i]
    num_subfigures = max_i  # from left to right
    num_nested_subfigures = max_j  # from top to bottom
    dataframe = dataframe_list[i]

    cols = config.plot.x_axis[i]
    idx = config.plot.y_axis[0]
    # only include colorbar once
    include_cbar: bool = i == num_subfigures - 1

    model_param = name_for_additional_subplots[j]
    if model_param == "default":
        current_dataframe = dataframe  # we do not need to do further grouping
    else:
        param_name, param_value = model_param.split("=", maxsplit=1)
        if pd.api.types.is_numeric_dtype(dataframe[param_name]):
            param_value = float(param_value)
        try:
            current_dataframe = dataframe.groupby([param_name]).get_group(
                (param_value,)
            )
        except KeyError:
            log_warn(
                f"WARNING: was not able to groupby '{param_name}',"
                + "maybe the data was created with different versions of fob; skipping this row"
            )
            log_debug(
                f"{param_name=}{param_value=}{dataframe.columns=}{dataframe[param_name]=}"
            )
            return False
    current_plot = create_matrix_plot(
        current_dataframe,
        config,
        cols,
        idx,
        ax=axs[j][i],
        cbar=include_cbar,
        vmin=vmin,
        vmax=vmax,
    )

    # LABELS
    # Pretty name for label "learning_rate" => "Learning Rate"
    # remove x_label of all but last row, remove y_label for all but first column
    if i > 0:
        current_plot.set_ylabel("", labelpad=8)
    else:
        current_plot.set_ylabel(pretty_name(current_plot.get_ylabel()))
    if j < num_nested_subfigures - 1:
        current_plot.set_xlabel("", labelpad=8)
    else:
        current_plot.set_xlabel(pretty_name(current_plot.get_xlabel()))

    # reading optimizer and task name after grouping
    df_entry = current_dataframe.iloc[0]  # just get an arbitrary trial
    opti_name = df_entry["optimizer.name"]
    task_name = df_entry["task.name"]

    # TITLE
    # title (heading) of the heatmap: <optimname> on <taskname> (+ additional info)
    title = f"{pretty_name(opti_name)} on {pretty_name(task_name)}"
    if max_i > 1 or max_j > 1:
        title += "" if model_param == "default" else f"\n{model_param}"
    current_plot.set_title(title)


def extract_dataframes(
    workload_paths: List[Path], config: AttributeDict, depth: int = 1
) -> list[pd.DataFrame]:
    df_list: list[pd.DataFrame] = []
    num_dataframes: int = len(workload_paths)

    for i in range(num_dataframes):
        available_trials = get_available_trials(workload_paths[i], config, depth)
        dataframe = dataframe_from_trials(available_trials, config)
        df_list.append(dataframe)

    return df_list


def get_output_file_path(
    dataframe_list: list[pd.DataFrame], config: AttributeDict, suffix: str = ""
) -> Path:
    task_names = [df.iloc[0]["task.name"] for df in dataframe_list]
    optim_names = [df.iloc[0]["optimizer.name"] for df in dataframe_list]
    task_name = "_".join(sorted(set(task_names)))
    optim_name = "_".join(sorted(set(optim_names)))

    here = Path(__file__).parent.resolve()

    output_dir = Path(config.output_dir) if config.output_dir else here
    experiment_name = (
        Path(config.experiment_name)
        if config.experiment_name
        else f"{optim_name}-{task_name}"
    )
    output_file_path = output_dir / experiment_name

    return Path(f"{output_file_path}-{suffix}" if suffix else output_file_path)


def set_plotstyle(config: AttributeDict):
    plt.rcParams["text.usetex"] = config.plotstyle.text.usetex
    plt.rcParams["font.family"] = config.plotstyle.font.family
    plt.rcParams["font.size"] = config.plotstyle.font.size


def pretty_name(
    name: str, pretty_names: dict | str = {}  # type: ignore pylint: disable=dangerous-default-value
) -> str:
    """
    Tries to use a mapping for the name, else will do some general replacement.
    mapping can be a directory or a filename of a yaml file with 'names' key
    """

    # reading from yaml and caching the dictionary
    label_file: Path = evaluation_path() / "labels.yaml"
    if isinstance(pretty_names, str):
        label_file = Path(pretty_names)

    if pretty_names == {} or isinstance(pretty_names, str):
        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(label_file)
        pretty_names: dict[str, str] = yaml_content["names"]

    # applying pretty names
    name_without_yaml_prefix = name.split(".")[-1]
    if name in pretty_names.keys():
        name = pretty_names[name]
    elif name_without_yaml_prefix in pretty_names.keys():
        name = pretty_names[name_without_yaml_prefix]
    else:
        name = name.replace("_", " ").title()
    return name


def save_csv(dfs: list[pd.DataFrame], output_filename: Path):
    for i, df in enumerate(dfs):
        csv_output_filename = f"{output_filename.resolve()}-{i}.csv"
        log_info(f"saving raw data as {csv_output_filename}")
        df.to_csv(path_or_buf=csv_output_filename, index=False)


def save_plot(fig: Figure, output_file_path: Path, file_type: str, dpi: int):
    plot_output_filename = f"{output_file_path.resolve()}.{file_type}"
    log_info(f"saving figure as <{plot_output_filename}>")
    fig.savefig(plot_output_filename, dpi=dpi)


def save_files(
    fig, dfs: list[pd.DataFrame], output_file_path: Path, config: AttributeDict
):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    for file_type in config.output_types:
        if file_type == "csv":
            save_csv(dfs, output_file_path)
        elif file_type == "png" or file_type == "pdf":
            save_plot(fig, output_file_path, file_type, config.plotstyle.dpi)


def clean_config(config: AttributeDict) -> AttributeDict:
    """some processing that allows the user to be lazy, shortcut for the namespace,
    hidden values are found and config.all_values"""
    if "evaluation" in config.keys():
        evaluation_config: AttributeDict = config.evaluation
        evaluation_config["all_values"] = config
        config = evaluation_config
    else:
        log_warn("there is no 'evaluation' in the yaml provided!")
    if "data_dirs" in config.keys():
        value_is_none = not config.data_dirs
        value_has_wrong_type = not isinstance(config.data_dirs, (PathLike, str, list))
        if value_is_none or value_has_wrong_type:
            raise ValueError(
                f"Error: 'evaluation.data_dirs' was not provided correctly! "
                f"check for typos in the yaml provided! value given: {config.data_dirs}"
            )

    # allow the user to write a single string instead of a list of strings
    if not isinstance(config.output_types, list):
        config["output_types"] = [config.output_types]
        log_info("fixing value for key <config.output_types> to be a list[str]")

    if not isinstance(config.data_dirs, list):
        config["data_dirs"] = [Path(config.data_dirs)]
        log_info("fixing value for key <config.data_dirs> to be a list[Path]")

    # x_axis
    if not isinstance(config.plot.x_axis, list):
        config["plot"]["x_axis"] = [config.plot.x_axis]
        log_info("fixing value for key <config.plot.x_axis> to be a list[str]")
    if len(config.plot.x_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.x_axis)
        config["plot"]["x_axis"] += repeat(config.plot.x_axis[0], missing_elements)

    # y_axis
    if not isinstance(config.plot.y_axis, list):
        config["plot"]["y_axis"] = [config.plot.y_axis]
        log_info("fixing value for key <config.plot.y_axis> to be a list[str]")
    if len(config.plot.y_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.y_axis)
        config["plot"]["y_axis"] += repeat(config.plot.y_axis[0], missing_elements)

    return config


def main(config: AttributeDict):
    config = clean_config(config)  # sets config to config.evaluation, cleans some data
    workloads: List[Path] = [Path(name) for name in config.data_dirs]
    log_debug(f"{workloads}=")

    set_plotstyle(config)

    dfs = extract_dataframes(workloads, depth=config.depth, config=config)
    fig, _ = create_figure(dfs, config)

    output_file_path = get_output_file_path(dfs, config)

    save_files(fig, dfs, output_file_path, config)
