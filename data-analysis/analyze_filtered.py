import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import causalpy as cp
from tqdm import tqdm
from core.paneldata import MAX_TIME_INTERVALS

plot_path = 'data/plots-filtered'

temporal_panel1 = pd.read_csv("data/sample500000filtered/sample500000-temporal_panel1.csv")
temporal_panel2 = pd.read_csv("data/sample500000/sample500000-temporal_panel2.csv")
user_panel = pd.read_csv("data/sample500000filtered/sample500000-user_panel.csv")

gpt35_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 0].sort_values("time")
gpt4_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 1].sort_values("time")
both_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 2].sort_values("time")

gpt35_kinks = []
gpt4_kinks = []

def make_plots(diversity_series1, name1, diversity_series2, name2):
    
    diversity_series1 = diversity_series1.dropna(subset=["concept_diversity_user_filtered"])
    diversity_series2 = diversity_series2.dropna(subset=["concept_diversity_user_filtered"])
    
    def smooth_curve(y: pd.Series):
        # Apply an exponential moving average to the y values
        return y.ewm(halflife=5).mean()
    
    # Enlarge plot size
    plt.figure(figsize=(10, 7.5))
    
    # Plot diversity curves
    # plt.plot(diversity_series1.time, diversity_series1.concept_diversity_filtered, label=f"{name1} (both)", color="blue", linestyle="-", markersize=3)
    plt.plot(diversity_series1.time, smooth_curve(diversity_series1.concept_diversity_user_filtered), label=f"{name1} (user)", color="blue", linestyle="-", marker=None)
    plt.scatter(diversity_series1.time, diversity_series1.concept_diversity_user_filtered, label=f"  ", color="blue")
    # plt.plot(diversity_series1.time, smooth_curve(diversity_series1.concept_diversity_assistant_filtered), label=f"{name1} (assistant)", color="blue", linestyle="--", marker=None)
    
    # Plot linear regression curve
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(diversity_series1.time.values.reshape(-1, 1), diversity_series1.concept_diversity_user_filtered.values.reshape(-1, 1))
    plt.plot(diversity_series1.time, model.predict(diversity_series1.time.values.reshape(-1, 1)), label=f"{name1} linear regression (user)", color="blue", linestyle="dotted", marker=None)
    
    # Calculate p value of the slope of the linear regression
    from statsmodels.formula.api import ols
    model = ols(f"concept_diversity_user_filtered ~ time", data=diversity_series1).fit()
    print(f"{name1} linear regression p-value: {model.pvalues['time']}")
    print(f"Full results:\n{model.summary()}")
    
    # Draw vertical line to indicate when version shifts within the GPT-3.5 family happened
    if '+' not in name1:
        rows_sorted = [row for _, row in diversity_series1.iterrows()]
        labeled = False
        for row_index, row in enumerate(rows_sorted):
            if row_index == 0:
                continue
            
            if row.gpt_version != rows_sorted[row_index - 1].gpt_version:
                plt.axvline(x=(row.time + rows_sorted[row_index - 1].time) / 2, color="blue", linestyle="dashdot", linewidth=3, label=(f"{name1} version update" if not labeled else None))
                gpt35_kinks.append((row.time + rows_sorted[row_index - 1].time) / 2)
                labeled = True

    # Plot diversity curves
    # plt.plot(diversity_series2.time, diversity_series2.concept_diversity_filtered, label=f"{name2} (both)", color="red", linestyle="-", markersize=3)
    plt.plot(diversity_series2.time, smooth_curve(diversity_series2.concept_diversity_user_filtered), label=f"{name2} (user)", color="red", linestyle="-", marker=None)
    plt.scatter(diversity_series2.time, diversity_series2.concept_diversity_user_filtered, label=f"  ", color="red")
    # plt.plot(diversity_series2.time, smooth_curve(diversity_series2.concept_diversity_assistant_filtered), label=f"{name2} (assistant)", color="red", linestyle="--", marker=None)
    
    # Plot linear regression curve
    model = LinearRegression()
    model.fit(diversity_series2.time.values.reshape(-1, 1), diversity_series2.concept_diversity_user_filtered.values.reshape(-1, 1))
    plt.plot(diversity_series2.time, model.predict(diversity_series2.time.values.reshape(-1, 1)), label=f"{name2} linear regression (user)", color="red", linestyle="dotted", marker=None)
    
    # Calculate p value of the slope of the linear regression
    model = ols(f"concept_diversity_user_filtered ~ time", data=diversity_series2).fit()
    print(f"{name2} linear regression p-value: {model.pvalues['time']}")
    print(f"Full results:\n{model.summary()}")
    
    # Draw vertical line to indicate when version shifts within the GPT-3.5 family happened
    if '+' not in name2:
        rows_sorted = [row for _, row in diversity_series2.iterrows()]
        labeled = False
        for row_index, row in enumerate(rows_sorted):
            if row_index == 0:
                continue
            
            if row.gpt_version != rows_sorted[row_index - 1].gpt_version:
                plt.axvline(x=(row.time + rows_sorted[row_index - 1].time) / 2, color="red", linestyle="dashdot", linewidth=3, label=(f"{name2} version update" if not labeled else None))
                gpt4_kinks.append((row.time + rows_sorted[row_index - 1].time) / 2)
                labeled = True
    
    
    plt.xlabel("Time")
    plt.ylabel("Diversity")
    plt.title("Diversity over time")
    plt.legend()
    plt.show()
    plt.savefig(f"{plot_path}/diversity_over_time_{name1.replace('+','')}_vs_{name2.replace('+','')}.pdf")

def rkd_regression_plot(df: pd.DataFrame, kink: float, name: str, seed=42, linear=False):
    import arviz as az
    rng = np.random.default_rng(seed)
    
    df["treated"] = (df.time >= kink).astype(int)
    df["x"] = df.time
    df["y"] = df.concept_diversity_user_filtered
    if linear:
        formula = f"y ~ 1 + x + I((x-{kink})*treated)"
    else:
        formula = f"y ~ 1 + x + np.power(x, 2) + I((x-{kink})*treated) + I(np.power(x-{kink}, 2)*treated)"

    result2 = cp.RegressionKink(
        df,
        formula=formula,
        model=cp.pymc_models.LinearRegression(sample_kwargs={"random_seed": seed}),
        kink_point=kink,
        epsilon=1.5,
    )
    fig, ax = result2.plot()
    fig.savefig(f"{plot_path}/rkd_regression_plot_{name}.pdf")
    print(result2.summary())
    
    # Clean up plot
    plt.clf()
    
    ax = az.plot_posterior(result2.gradient_change, ref_val=0)
    ax.set(title="Gradient change")
    plt.show()
    plt.savefig(f"{plot_path}/rkd_regression_plot_{name}_bayesian.pdf")
    

def its_test(df: pd.DataFrame, kink: float, name: str, gaussian=False):
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
    
    df["treated"] = (df.time >= kink).astype(int)
    df["t"] = df.time
    df["y"] = df.concept_diversity_user_filtered
    
    # The best multiple of 7 to use for the seasonal component (the one with highest R2), determined for each kink separately
    df["weekday"] = df.time % (7 if kink <= 40 else 21 if "35" in name else 21)
    
    # Remove rows with missing values in the columns we need
    df = df.dropna(subset=["treated", "t", "y", "weekday"])
    
    # if "time" isn't df's index, set it as the index
    if df.index.name != "time":
        df = df.set_index("time")
    
    # kernel = RBF() + WhiteKernel()
    kernel = 1.0 * ExpSineSquared(10.0, 28.0) + WhiteKernel(.01)
    result = cp.InterruptedTimeSeries(
        df,
        kink,
        formula="y ~ 1 + t + C(weekday)",
        model=(LinearRegression() if not gaussian else GaussianProcessRegressor(kernel=kernel, normalize_y=True)),
    )
    
    fig, ax = result.plot()
    plt.show()
    plt.savefig(f"{plot_path}/its_test_{name}.pdf")
    
    # fig, ax = result.bayesian_plot()
    # plt.show()
    # plt.savefig(f"data/bayesian/its_test_{name}_bayesian.pdf")
    
    print(result.summary(round_to=4))


def rdd_regression_plot(df: pd.DataFrame, kink: float, name: str, seed=42, linear=False):
    import arviz as az
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
    from sklearn.gaussian_process import GaussianProcessRegressor

    rng = np.random.default_rng(seed)
    
    df["treated"] = (df.time >= kink).astype(int)
    df["x"] = df.time
    df["y"] = df.concept_diversity_user
    if linear:
        formula = "y ~ 1 + x + treated"
        model = LinearRegression()
        bandwidth = 40
    else:
        formula = "y ~ 1 + bs(x, df=4) + treated"
        model = LinearRegression()
        bandwidth = np.inf
        # formula = "y ~ 1 + x + treated"
        # # kernel = RBF(length_scale_bounds=[1e-9, 1e18]) + WhiteKernel()
        # kernel = WhiteKernel(1e-1)
        # model = GaussianProcessRegressor(kernel=kernel)
        # bandwidth = np.inf

    result2 = cp.RegressionDiscontinuity(
        df,
        formula=formula,
        model=model,
        treatment_threshold=kink,
        bandwidth=bandwidth,
    )
    fig, ax = result2.plot()
    plt.show()
    plt.savefig(f"{plot_path}/rdd_regression_plot_{name}.pdf")
    try:
        print(result2.summary())
    except AttributeError as e:
        print(e)


def process_user_panel(df: pd.DataFrame, y_variable: str, family: int = None):
    
    # Visualize distribution of df.nsamples
    ax = df.nsamples.plot.hist(bins=1000)
    
    # Use log-log scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.show()
    plt.savefig(f"{plot_path}/user_panel_nsamples_histogram.pdf")
    
    # Calculate earliest entry date by applying min_func to nsamples_temporal_composition
    def min_func(tp: tuple):
        tp = list(eval(tp))
        assert 100 <= len(tp) <= 150
        for i, v in enumerate(tp):
            if v:
                return i
        
        assert False, "No nonzero value found"
    
    df["user_first_entry"] = df.nsamples_temporal_composition.apply(min_func)
    
    # Calculate model family use freq by applying div_func to nsamples_version_composition
    def div_func(tp: tuple):
        tp = list(eval(tp))
        if not (len(tp) == 6):
            print(tp, len(tp))
            assert len(tp) == 6
        
        return sum(tp[:3]) / sum(tp)
    
    df["user_gpt35_ratio"] = df.nsamples_version_composition.apply(div_func)
    
    if family is not None:
        target_ratio = 1 - family
        df = df[df.user_gpt35_ratio >= target_ratio - 0.2]
        df = df[df.user_gpt35_ratio <= target_ratio + 0.2]
    
    # Reduce location to first element (nation)
    df["location"] = df.location.apply(lambda x: eval(x)[0])
    
    # Add interaction term between user_first_entry and temporal_extension
    df["user_first_entry_X_temporal_extension"] = df.user_first_entry * df.temporal_extension
    
    if y_variable == "diversity_loss":
        def diversity_loss(tp: tuple):
            if isinstance(tp, str):
                tp = eval(tp.replace('nan', 'None'))
            
            if not isinstance(tp, tuple):
                print(tp)
            assert isinstance(tp, tuple)
            tp = list(tp)
            assert 100 <= len(tp) <= 150
            
            tp = [v for v in tp if v is not None and v != np.nan]
            
            if len(tp) == 0:
                return None
            
            return tp[0] - tp[-1]
        
        df["y"] = df.concept_diversity_across_time_filtered.apply(diversity_loss)
    else:
        df["y"] = df[y_variable]
    
    return df


def user_regression(df: pd.DataFrame, y_variable: str, family: int = None):
    import statsmodels.api as sm
    
    # Extract independent variables
    X = df[["user_gpt35_ratio", "language", "location", "nsamples", "temporal_extension", "user_first_entry", "user_first_entry_X_temporal_extension", "mean_turns", "mean_conversation_length", "mean_prompt_length"]]
    X = pd.get_dummies(X, columns=["language", "location"], drop_first=True, dtype=int)
    
    # Add constant term to independent variables
    X = sm.add_constant(X)
    
    # Extract dependent variable
    y = df["y"]
    
    # Fit OLS model
    X.to_csv(f"{plot_path}/user_regression_X.csv")
    y.to_csv(f"{plot_path}/user_regression_y_{y_variable + (str(family) if family is not None else '')}.csv")
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    print(results.summary())
    
    # Perform test for heteroskedasticity
    _, pval, _, _ = sm.stats.diagnostic.het_breuschpagan(results.resid, results.model.exog)
    print(f"Breusch-Pagan test p-value: {pval}")
    if pval < 0.05:
        print("Heteroskedasticity detected")
        robust_results = results.get_robustcov_results(cov_type="HC3")
        print(robust_results.summary())


def user_temporal_regression(user_panel: pd.DataFrame, y_variable: str, family: int = None):
    import statsmodels.api as sm
    
    # Transform string to list
    df = user_panel.copy()
    df["concept_diversity_across_time"] = df.concept_diversity_across_time.apply(lambda x: list(eval(x.replace("nan", "None"))))
    
    # Calculate the length of concept_diversity_across_time
    shared_length = df.concept_diversity_across_time.apply(len).min()
    assert 100 <= shared_length <= 150
    assert df.concept_diversity_across_time.apply(len).max() == shared_length
    
    # Add indexing to concept_diversity_across_time
    df["time"] = df.concept_diversity_across_time.apply(lambda x: list(range(len(x))))
    
    def nonnull_count(tp: list):
        nonnull = [int(v is not None and v != np.nan) for v in tp]
        if sum(nonnull) <= 4:
            return [None] * len(nonnull)
        prefix_sum = np.cumsum(nonnull) / sum(nonnull)
        return prefix_sum
    
    df["progress"] = df.concept_diversity_across_time.apply(nonnull_count)
    
    # Expand each element of concept_diversity_across_time into separate rows
    df = df.explode(["concept_diversity_across_time", "time", "progress"])
    df["progress"] = df.progress.astype(float)
    df["time"] = df.time.astype(int)
    df["user_id"] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["concept_diversity_across_time", "time", "progress"])
    
    # Extract independent variables
    X = df[["user_gpt35_ratio", "language", "location", "nsamples", "temporal_extension", "version_diversity", "user_first_entry", "mean_turns", "mean_conversation_length", "mean_prompt_length", "progress", "time"]]
    X = pd.get_dummies(X, columns=["language", "location"], drop_first=True, dtype=int)
    
    # Add constant term to independent variables
    X = sm.add_constant(X)
    
    # Extract dependent variable
    y = df["concept_diversity_across_time"].astype(float)
    
    # Fit OLS model
    X.to_csv(f"{plot_path}/user_temporal_regression_X.csv")
    y.to_csv(f"{plot_path}/user_temporal_regression_y_{y_variable + (str(family) if family is not None else '')}.csv")
    print(X.info(max_cols=200))
    print(y.info())
    model = sm.MixedLM(y, X, groups=df.user_id, missing="drop")
    results = model.fit()
    print(results.summary())
    
    # Use hac-groupsum
    X["user_id"] = df.user_id
    X = pd.get_dummies(X, columns=["user_id"], drop_first=True, dtype=int)
    linear_results = sm.OLS(y, X, missing="drop").fit()
    robust_results = linear_results.get_robustcov_results(cov_type="hac-groupsum", time=X["time"].to_numpy(), maxlags=2)
    print('\n'.join(robust_results.summary().as_text().splitlines()[:100]))


def visualize_tree(threshold=250, leading=15, skip_banned: bool = True):
    from core.paneldata import BANNED_CLUSTERS
    from utils.json_utils import load_file
    clusterization_results = load_file("sample500000/sample500000-clusterinfo-postselection.json")
    selected_clusters = clusterization_results["selected_clusters"]
    cluster_selected_parent = clusterization_results["cluster_selected_parent"]
    cluster_name = clusterization_results["cluster_name"]
    cluster_size = clusterization_results["cluster_size"]
    root = clusterization_results["root"]
    
    def is_banned(concept: int) -> bool:
        if not skip_banned:
            return False
        
        while concept:
            if concept in BANNED_CLUSTERS:
                return True
            concept = cluster_selected_parent[concept]
        return False
    
    from treelib import Node, Tree

    tree = Tree()
    
    selected_clusters = sorted(selected_clusters + [root], key=lambda x: cluster_size[x], reverse=True)
    print(selected_clusters[:5])
    print([cluster_size[id] for id in selected_clusters[:5]])
    cluster_selected_parent[root] = None
    for id in tqdm(selected_clusters):
        if cluster_size[id] < threshold:
            continue
        tree.create_node(cluster_name[id], id, parent=(root if id != root else None))
    
    for id in tqdm(selected_clusters):
        if cluster_size[id] < threshold:
            continue
        if id != root:
            tree.move_node(id, cluster_selected_parent[id])
    
    tree.save2file(f"{plot_path}/tree.txt")
    
    concepts_df = temporal_panel2.groupby(["cluster", "is_gpt4"], as_index=False).cluster_nsamples.sum().sort_values("cluster_nsamples", ascending=False).reset_index()
    print(concepts_df.head(20))
    print(concepts_df.describe())
    
    print("GPT-4 Stats")
    print(concepts_df[concepts_df.is_gpt4 == 1])
    print(concepts_df[concepts_df.is_gpt4 == 1].describe())
    
    agg_concepts_df = concepts_df.copy()
    agg_concepts_df["contribution"] = (-1) ** agg_concepts_df.is_gpt4 * np.log(agg_concepts_df.cluster_nsamples + 1)
    agg_concepts_df = agg_concepts_df.groupby(["cluster"]).sum().sort_values("contribution", ascending=False).reset_index()
    print(agg_concepts_df)
    print(agg_concepts_df.describe())
    
    # Get the nsamples
    print("Temporal Panel2")
    temporal_panel2.set_index(["time", "is_gpt4", "cluster"], inplace=True)
    print(temporal_panel2)
    print(temporal_panel2.loc[(118, 0, 434978), "cluster_nsamples"])
    gpt35_df = concepts_df[concepts_df.is_gpt4 == 0]
    gpt4_df = concepts_df[concepts_df.is_gpt4 == 1]
    
    def get_freq(id, time, family):
        index_vals = (time, family, id)
        try:
            return temporal_panel2.loc[index_vals, "cluster_nsamples"]
        except:
            return 0
    
    def print_cluster_info(id):
        gpt4_nsamples = gpt4_df[gpt4_df.cluster == id].cluster_nsamples.sum()
        gpt35_nsamples = gpt35_df[gpt35_df.cluster == id].cluster_nsamples.sum()
        temporal_freqs_gpt4 = [get_freq(id, time, 1) for time in range(MAX_TIME_INTERVALS)]
        temporal_freqs_gpt35 = [get_freq(id, time, 0) for time in range(MAX_TIME_INTERVALS)]
        return f"Cluster {id} {cluster_size[id]:07d} [{(gpt35_nsamples+1) / (gpt4_nsamples+1):7.07f} = {gpt35_nsamples:07d} (GPT3.5) / {gpt4_nsamples:07d} (GPT4)]: {cluster_name[id]}\n" + \
               "GPT35: " + " ".join([f"{v:07d}" for v in temporal_freqs_gpt35]) + "\n" + \
               "GPT4: " + " ".join([f"{v:07d}" for v in temporal_freqs_gpt4]) + "\n\n"
    
    cluster_list = agg_concepts_df.cluster.tolist()
    answer = ""
    for id in tqdm(cluster_list):
        if cluster_size[id] < threshold or id not in selected_clusters or is_banned(id):
            continue
        
        answer += print_cluster_info(id) + "\n"
    
    with open(f"{plot_path}/ranked_nodes.txt", "w") as f:
        f.write(answer)


if __name__ == "__main__":
    # make_plots(gpt35_diversity_series, "GPT3.5-turbo", both_diversity_series, "GPT3.5-turbo+GPT4")
    # make_plots(gpt35_diversity_series, "GPT3.5-turbo", gpt4_diversity_series, "GPT4")
    
    # rkd_regression_plot(gpt35_diversity_series, gpt35_kinks[0], "gpt35_kink1_linear", linear=True)
    # rkd_regression_plot(gpt35_diversity_series, gpt35_kinks[1], "gpt35_kink2_linear", linear=True)
    # rkd_regression_plot(gpt4_diversity_series, gpt4_kinks[1], "gpt4_kink2_linear", linear=True)
    # rkd_regression_plot(both_diversity_series, gpt35_kinks[0], "total_gpt35_kink1_linear", linear=True)
    # rkd_regression_plot(both_diversity_series, gpt35_kinks[1], "total_gpt35_kink2_linear", linear=True)
    
    # its_test(gpt35_diversity_series, gpt35_kinks[0], "gpt35_kink1", gaussian=False)
    # its_test(gpt35_diversity_series, gpt35_kinks[1], "gpt35_kink2", gaussian=False)
    # its_test(gpt4_diversity_series, gpt4_kinks[1], "gpt4_kink2", gaussian=False)
    # its_test(both_diversity_series, gpt35_kinks[0], "total_gpt35_kink1", gaussian=False)
    # its_test(both_diversity_series, gpt35_kinks[1], "total_gpt35_kink2", gaussian=False)
    # its_test(both_diversity_series, gpt4_kinks[1], "total_gpt4_kink2", gaussian=False)
    
    # rdd_regression_plot(gpt35_diversity_series, gpt35_kinks[0], "gpt35_kink1_linear", linear=True)
    # rdd_regression_plot(gpt35_diversity_series, gpt35_kinks[1], "gpt35_kink2_linear", linear=True)
    # rdd_regression_plot(gpt4_diversity_series, gpt4_kinks[1], "gpt4_kink2_linear", linear=True)
    # rdd_regression_plot(gpt35_diversity_series, gpt35_kinks[0], "gpt35_kink1_Bspline", linear=False)
    # rdd_regression_plot(gpt35_diversity_series, gpt35_kinks[1], "gpt35_kink2_Bspline", linear=False)
    # rdd_regression_plot(gpt4_diversity_series, gpt4_kinks[1], "gpt4_kink2_Bspline", linear=False)
    
    # y_variable, family = "concept_diversity_filtered", 1
    # user_panel = process_user_panel(user_panel, y_variable=y_variable, family=family)
    # user_regression(user_panel, y_variable=y_variable, family=family)
    # user_temporal_regression(user_panel, y_variable=y_variable, family=family)
    
    visualize_tree()