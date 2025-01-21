import os, sys
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import causalpy as cp
from tqdm import tqdm
from core.paneldata import MAX_TIME_INTERVALS
import math

plot_path = 'data/fullrec-plots-unfiltered'

temporal_panel1 = pd.read_csv("data/fullrec/b41fd1b7bdad96e440d15a97d640827e-temporal_panel1.csv")
# temporal_panel2 = pd.read_csv("data/sample500000/sample500000-temporal_panel2.csv")
user_panel = pd.read_csv("data/fullrec/b41fd1b7bdad96e440d15a97d640827e-user_panel.csv")

gpt35_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 0].sort_values("time")
gpt4_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 1].sort_values("time")
both_diversity_series = temporal_panel1[temporal_panel1.is_gpt4 == 2].sort_values("time")

gpt35_kinks = []
gpt4_kinks = []

def make_plots(diversity_series1, name1, diversity_series2, name2):
    
    diversity_series1 = diversity_series1.dropna(subset=["concept_diversity_user"])
    diversity_series2 = diversity_series2.dropna(subset=["concept_diversity_user"])
    
    def smooth_curve(y: pd.Series):
        # Apply an exponential moving average to the y values
        return y.ewm(halflife=5).mean()
    
    # Enlarge plot size
    plt.figure(figsize=(10, 7.5))
    
    # Plot diversity curves
    # plt.plot(diversity_series1.time, diversity_series1.concept_diversity, label=f"{name1} (both)", color="blue", linestyle="-", markersize=3)
    plt.plot(diversity_series1.time, smooth_curve(diversity_series1.concept_diversity_user), label=f"{name1} (smoothed diversity curve)", color="blue", linestyle="-", marker=None)
    plt.scatter(diversity_series1.time, diversity_series1.concept_diversity_user, color="blue")
    # plt.plot(diversity_series1.time, smooth_curve(diversity_series1.concept_diversity_assistant), label=f"{name1} (assistant)", color="blue", linestyle="--", marker=None)
    
    # Plot linear regression curve
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(diversity_series1.time.values.reshape(-1, 1), diversity_series1.concept_diversity_user.values.reshape(-1, 1))
    plt.plot(diversity_series1.time, model.predict(diversity_series1.time.values.reshape(-1, 1)), label=f"{name1} (linear regression)", color="blue", linestyle="dotted", marker=None)
    
    # Calculate p value of the slope of the linear regression
    from statsmodels.formula.api import ols
    model = ols(f"concept_diversity_user ~ time", data=diversity_series1).fit()
    print(f"{name1} linear regression p-value: {model.pvalues['time']}")
    print(f"Full results:\n{model.summary()}")
    
    # Plot quadratic regression curve
    # model = LinearRegression()
    # input_x = np.column_stack((diversity_series1.time.values.reshape(-1, 1), np.power(diversity_series1.time.values.reshape(-1, 1), 2)))
    # model.fit(input_x, diversity_series1.concept_diversity_user.values.reshape(-1, 1))
    # plt.plot(diversity_series1.time, model.predict(input_x), label=f"{name1} (quadratic regression)", color="blue", linestyle="dotted", marker=None)
    
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
    # plt.plot(diversity_series2.time, diversity_series2.concept_diversity, label=f"{name2} (both)", color="red", linestyle="-", markersize=3)
    plt.plot(diversity_series2.time, smooth_curve(diversity_series2.concept_diversity_user), label=f"{name2} (smoothed diversity curve)", color="red", linestyle="-", marker=None)
    plt.scatter(diversity_series2.time, diversity_series2.concept_diversity_user, color="red")
    # plt.plot(diversity_series2.time, smooth_curve(diversity_series2.concept_diversity_assistant), label=f"{name2} (assistant)", color="red", linestyle="--", marker=None)
    
    # Plot linear regression curve
    model = LinearRegression()
    model.fit(diversity_series2.time.values.reshape(-1, 1), diversity_series2.concept_diversity_user.values.reshape(-1, 1))
    plt.plot(diversity_series2.time, model.predict(diversity_series2.time.values.reshape(-1, 1)), label=f"{name2} (linear regression)", color="red", linestyle="dotted", marker=None)
    
    # Calculate p value of the slope of the linear regression
    model = ols(f"concept_diversity_user ~ time", data=diversity_series2).fit()
    print(f"{name2} linear regression p-value: {model.pvalues['time']}")
    print(f"Full results:\n{model.summary()}")
    
    # Plot quadratic regression curve
    # model = LinearRegression()
    # input_x = np.column_stack((diversity_series2.time.values.reshape(-1, 1), diversity_series2.time.values.reshape(-1, 1) ** 2))
    # model.fit(input_x, diversity_series2.concept_diversity_user.values.reshape(-1, 1))
    # plt.plot(diversity_series2.time, model.predict(input_x), label=f"{name2} (quadratic regression)", color="red", linestyle="dotted", marker=None)
    
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
    plt.ylabel("Conceptual Diversity in Human Messages")
    plt.title("Diversity over Time (Apr 2023 to Apr 2024)")
    plt.ylim((2.52,2.59))
    plt.legend()
    plt.show()
    plt.savefig(f"{plot_path}/diversity_over_time_{name1.replace('+','')}_vs_{name2.replace('+','')}.pdf")

def rkd_regression_plot(df: pd.DataFrame, kink: float, name: str, seed=42, linear=False):
    import arviz as az
    rng = np.random.default_rng(seed)
    
    df["treated"] = (df.time >= kink).astype(int)
    df["x"] = df.time
    df["y"] = df.concept_diversity_user
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
    df["y"] = df.concept_diversity_user
    
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


def process_user_panel(df: pd.DataFrame, y_variable: str, family: int = None, truncate_rate=1):
    
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
        
        df["y"] = df.concept_diversity_across_time.apply(diversity_loss)
    else:
        df["y"] = df[y_variable]
    
    # Remove abnormally large nsamples
    # df = df[df.nsamples < 1000]
    
    # Truncate the dataset, leaving only the top truncate_rate percent of users by nsamples
    df = df.sort_values("nsamples", ascending=False)
    df = df.head(int(truncate_rate * len(df)))
    print(f"Truncated dataset to {truncate_rate * 100}% of users")
    print(df.describe())
    
    # df["nsamples"] = np.log(df.nsamples)
    
    return df


def user_regression(df: pd.DataFrame, y_variable: str, family: int = None, truncate_rate=1):
    import statsmodels.api as sm
    
    df["nsamples_squared"] = df.nsamples ** 2
    
    # Extract independent variables
    columns = ["nsamples"]
    columns += ["user_first_entry"]
    columns += ["temporal_extension"]
    columns += ["user_gpt35_ratio"]
    columns += ["mean_turns", "mean_conversation_length", "mean_prompt_length"]
    columns += ["user_first_entry_X_temporal_extension"]
    # columns = ["nsamples_squared"] + columns
    control_columns = []
    control_columns += ["language"]
    # control_columns += ["location"]
    
    X = df[columns + control_columns]
    X = pd.get_dummies(X, columns=control_columns, drop_first=True, dtype=int)
    
    # Add constant term to independent variables
    X = sm.add_constant(X)
    
    # Extract dependent variable
    y = df["y"]
    
    # Fit OLS model
    X.to_csv(f"{plot_path}/user_regression_X_{int(math.log2(truncate_rate))}.csv")
    y.to_csv(f"{plot_path}/user_regression_y_{y_variable + (str(family) if family is not None else '')}_{int(math.log2(truncate_rate))}.csv")
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    print(results.summary())
    results_str = str(results.summary())
    
    # Perform test for heteroskedasticity
    _, pval, _, _ = sm.stats.diagnostic.het_breuschpagan(results.resid, results.model.exog)
    print(f"Breusch-Pagan test p-value: {pval}")
    results_str += f"\n\nBreusch-Pagan test p-value: {pval}"
    
    if pval < 0.05:
        print("Heteroskedasticity detected")
        robust_results = results.get_robustcov_results(cov_type="HC3")
        print(robust_results.summary())
        results_str += "\n\n" + str(robust_results.summary())
    
    with open(f"{plot_path}/user_regression_results_{y_variable + (str(family) if family is not None else '')}_{int(math.log2(truncate_rate))}.txt", "w") as f:
        f.write(results_str)


def user_temporal_regression(user_panel: pd.DataFrame, y_variable: str, family: int = None, truncate_rate=1, skip_reg=False) -> pd.DataFrame:
    import statsmodels.api as sm
    
    # Transform string to list
    df = user_panel.copy()
    df["concept_diversity_across_time"] = df.concept_diversity_across_time.apply(lambda x: list(eval(x.replace("nan", "None"))))
    df["nsamples_temporal_composition"] = df.nsamples_temporal_composition.apply(lambda x: list(eval(x)))
    
    # Calculate the length of concept_diversity_across_time
    shared_length = df.concept_diversity_across_time.apply(len).min()
    assert 100 <= shared_length <= 150
    assert df.concept_diversity_across_time.apply(len).max() == shared_length
    
    # Add indexing to concept_diversity_across_time
    df["time"] = df.concept_diversity_across_time.apply(lambda x: list(range(len(x))))
    
    # def nonnull_count(tp: list):
    #     nonnull = [int(v is not None and v != np.nan) for v in tp]
    #     if sum(nonnull) <= 4:
    #         return [None] * len(nonnull)
    #     prefix_sum = np.cumsum(nonnull) / sum(nonnull)
    #     return prefix_sum

    def engagement_progress(tp: list):
        assert len(tp) == shared_length
        return np.cumsum(tp)
    
    df["engagement_progress"] = df.nsamples_temporal_composition.apply(engagement_progress)
    
    # Expand each element of concept_diversity_across_time into separate rows
    df = df.explode(["concept_diversity_across_time", "time", "engagement_progress"])
    df["engagement_progress"] = df.engagement_progress.astype(float)
    df["time"] = df.time.astype(int)
    df["user_id"] = df.index
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["concept_diversity_across_time", "time", "engagement_progress"])

    df["eng_prog_squared"] = df.engagement_progress ** 2
    
    if skip_reg:
       return df 
    
    # Extract independent variables
    columns = ["nsamples"]
    columns += ["version_diversity", "user_first_entry", "engagement_progress"]
    columns += ["temporal_extension"]
    columns += ["user_gpt35_ratio"]
    columns += ["mean_turns", "mean_conversation_length", "mean_prompt_length"]
    # columns += ["eng_prog_squared"]
    
    control_columns = []
    control_columns += ["language"]
    # control_columns += ["location"]
    
    X = df[columns + control_columns]
    X = pd.get_dummies(X, columns=control_columns, drop_first=True, dtype=int)
    
    # Add constant term to independent variables
    X = sm.add_constant(X)
    
    # Extract dependent variable
    y = df["concept_diversity_across_time"].astype(float)
    
    # Fit OLS model
    X.to_csv(f"{plot_path}/user_temporal_regression_X_{int(math.log2(truncate_rate))}.csv")
    y.to_csv(f"{plot_path}/user_temporal_regression_y_{y_variable + (str(family) if family is not None else '')}_{int(math.log2(truncate_rate))}.csv")
    print(X.info(max_cols=200))
    print(y.info())
    model = sm.MixedLM(y, X, groups=df.user_id, missing="drop")
    results = model.fit()
    print(results.summary())
    results_str = str(results.summary())
    
    # Use hac-groupsum
    X["user_id"] = df.user_id
    X = pd.get_dummies(X, columns=["user_id"], drop_first=True, dtype=int)
    linear_results = sm.OLS(y, X, missing="drop").fit()
    robust_results = linear_results.get_robustcov_results(cov_type="hac-groupsum", time=df["time"].to_numpy(), maxlags=2)
    print('\n'.join(robust_results.summary().as_text().splitlines()[:100]))
    results_str += "\n\n" + "\n".join(robust_results.summary().as_text().splitlines()[:500])
    
    with open(f"{plot_path}/user_temporal_regression_results_{y_variable + (str(family) if family is not None else '')}_{int(math.log2(truncate_rate))}.txt", "w") as f:
        f.write(results_str)    
    
    return df


def user_diversity_plot(df: pd.DataFrame = user_panel, suffix = "", use_log = True):
    # X axis: log(nsamples)
    # Y axis: concept_diversity
    # Color: entry_date
    # Shape: language
    
    # Enlarge plot size
    plt.figure(figsize=(10, 7.5))
    
    # Plot diversity scatter plot
    if use_log:
        plt.scatter(df.nsamples, df.concept_diversity, c=df.user_first_entry*3, cmap="viridis", s=2, alpha=0.4)
    
    # Calculate binned means
    if use_log:
        lower_bins = np.arange(df.nsamples.min(), df.nsamples.min() * 10, dtype=float)
        upper_bins = np.logspace(np.log2(df.nsamples.min() * 11), np.log2(df.nsamples.max()), 100, base=2)
        bins = np.concatenate((lower_bins, upper_bins))
        df["bins"] = np.digitize(df.nsamples, bins)
        df["log_nsamples"] = np.log(df.nsamples)
        mean_diversity = df.groupby("bins").concept_diversity.mean()
        mean_nsamples = df.groupby("bins").log_nsamples.mean()
        plt.plot(np.exp(mean_nsamples), mean_diversity, label="Binned means", color="red", linestyle="-", marker="o", markersize=2)
    
    else:
        bins = np.linspace(df.nsamples.min(), df.nsamples.max(), 100)
        df["bins"] = np.digitize(df.nsamples, bins)
        df["log_nsamples"] = df.nsamples
        mean_diversity = df.groupby("bins").concept_diversity.mean()
        mean_nsamples = df.groupby("bins").nsamples.mean()
        plt.plot(mean_nsamples, mean_diversity, label="Binned means", color="red", linestyle="-", marker="o", markersize=2)
    
    # Make spline fit to the data
    # from sklearn.preprocessing import SplineTransformer
    # from sklearn.linear_model import LinearRegression
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import StandardScaler
    # reg = make_pipeline(StandardScaler(), SplineTransformer(n_knots=2, degree=2), LinearRegression())
    # reg.fit(df.nsamples.values.reshape(-1, 1), df.concept_diversity)
    # x = np.exp(np.linspace(np.log(df.nsamples.min()), np.log(df.nsamples.max()), 1000))
    # y = reg.predict(x.reshape(-1, 1))
    # plt.plot(x, y, label="Spline fit", color="red")
    
    # Make polynomial fit to the binning means
    x = mean_nsamples
    y = mean_diversity
    xnew = np.linspace(x.min(), x.max(), 1000)
    p = np.polyfit(x, y, 2)
    ynew = np.polyval(p, xnew)
    if use_log:
        plt.plot(np.exp(xnew), ynew, label="Polynomial fit", color="orange")
    else:
        plt.plot(xnew, ynew, label="Polynomial fit", color="orange")
    
    # Re-binning into 5 bins and plot means and error bars
    if use_log:
        bins = np.logspace(np.log2(df.nsamples.min()), np.log2(df.nsamples.max()), 9, base=2)
    else:
        bins = np.linspace(df.nsamples.min(), df.nsamples.max(), 9)
    
    df["bins"] = np.digitize(df.nsamples, bins)
    mean_diversity = df.groupby("bins").concept_diversity.mean()
    std_diversity = df.groupby("bins").concept_diversity.std()
    data_points = df.groupby("bins").concept_diversity.count()
    mean_nsamples = df.groupby("bins").log_nsamples.mean()
    if use_log:
        plt.errorbar(np.exp(mean_nsamples), mean_diversity, yerr=std_diversity / np.sqrt(data_points), label="Octile means", color="black", linestyle="None", marker="o", markersize=3)
    else:
        plt.errorbar(mean_nsamples, mean_diversity, yerr=std_diversity / np.sqrt(data_points), label="Octile means", color="black", linestyle="None", marker="o", markersize=3)
    
    # Add labels
    if use_log:
        plt.xlabel("Number of Conversations Participated In")
        plt.ylabel("Conceptual Diversity in Human Messages")
        plt.title("Diversity vs. Engagement")
    else:
        plt.xlabel("User Engagement Progress")
        plt.xticks([0,1], ["first message", "last message"], rotation=45)
        plt.ylabel("Conceptual Diversity in User Messages")
        plt.title("Diversity vs. Engagement")
    
    # Add colorbar with entry date legend
    if use_log:
        plt.colorbar(label="User Start Date (days since Apr 2023)")
    
    # Set log scale for x axis
    if use_log:
        plt.xscale("log")
        plt.ylim((2.2, 2.6))
    else:
        plt.xlim((0,1))
        mean_diversity_all = df.concept_diversity.mean()
        plt.ylim((mean_diversity_all - .02, mean_diversity_all + .04))
    
    # Show plot
    plt.legend(loc="lower right" if use_log else "upper right")
    plt.show()
    plt.savefig(f"{plot_path}/user_diversity_plot{'' if not suffix else '_' + suffix}.pdf")
    

def calculate_difference():
    from utils.json_utils import load_file
    clusterization_results = load_file("fullrec/b41fd1b7bdad96e440d15a97d640827e-clusterinfo-postselection.json")
    selected_clusters = clusterization_results["selected_clusters"]
    cluster_selected_parent = clusterization_results["cluster_selected_parent"]
    cluster_name = clusterization_results["cluster_name"]
    cluster_size = clusterization_results["cluster_size"]
    root = clusterization_results["root"]
    
    logdifs = []
    weights = []
    for id in tqdm(selected_clusters):
        if id == root:
            continue
        
        parent = cluster_selected_parent[id]
        logdifs.append(np.log2(cluster_size[parent]) - np.log(cluster_size[id]))
        weights.append(cluster_size[id])
    
    logdifs = np.array(logdifs)
    weights = np.array(weights)
    
    avg_logdif = np.average(logdifs)
    weighted_logdif = np.average(logdifs, weights=weights)
    logweighted_logdif = np.average(logdifs, weights=np.log(weights))
    print(f"Average log difference: {avg_logdif}")
    print(f"Weighted log difference: {weighted_logdif}")
    print(f"Log-weighted log difference: {logweighted_logdif}")


def visualize_tree(threshold=2500, leading=15, skip_banned: bool = False, compare_families: bool = False):
    from core.paneldata import BANNED_CLUSTERS
    from utils.json_utils import load_file
    clusterization_results = load_file("fullrec/b41fd1b7bdad96e440d15a97d640827e-clusterinfo-postselection.json")
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
    
    if root not in selected_clusters:
        selected_clusters.append(root)
    
    selected_clusters = sorted(selected_clusters, key=lambda x: cluster_size[x], reverse=True)
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
    
    tree.save2file(f"{plot_path}/tree.txt", key=lambda node: cluster_size[node.identifier], reverse=True)
    
    if not compare_families:
        return
    
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
    # rkd_regression_plot(both_diversity_series, gpt4_kinks[1], "total_gpt4_kink2_linear", linear=True)
    
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
    
    y_variable, family, truncate_rate = "concept_diversity", None, 1
    user_panel = process_user_panel(user_panel, y_variable=y_variable, family=family, truncate_rate=truncate_rate)
    # user_diversity_plot()
    # user_regression(user_panel, y_variable=y_variable, family=family, truncate_rate=truncate_rate)
    df = user_temporal_regression(user_panel, y_variable=y_variable, family=family, truncate_rate=truncate_rate, skip_reg=True)
    df.nsamples = df.engagement_progress / df.nsamples
    user_diversity_plot(df, "engagement_progress", use_log=False)
    
    # visualize_tree()
    # calculate_difference()