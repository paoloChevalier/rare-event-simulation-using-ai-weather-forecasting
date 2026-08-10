import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import curve_fit


def setup_latex_style(base_size=10):
    rcParams.update(
        {
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size - 1,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "contour.linewidth": 0.6,
        }
    )


def get_figsize(width_pt, fraction=1.0, height_factor=0.75):
    inches_per_pt = 1.0 / 72.27
    fig_width_in = width_pt * fraction * inches_per_pt
    fig_height_in = fig_width_in * height_factor
    return (fig_width_in, fig_height_in)


setup_latex_style()

WIDTH_PAPER = 372.0
WIDTH_INSA = 496.0


def check_chaos(
    tss, title, era=None, time_of_day=[0, 6, 12, 18], save=None, hline=None
):
    """Plot the trajectory of an ensemble forecast to evaluate chaotic spread.

    Extracts the top 3 highest temperature ensemble members, highlights them, 
    and shades the full ensemble spread alongside the ensemble mean. Can optionally 
    overlay a reference dataset (like ERA5) or a horizontal threshold line.

    Args:
        tss: xarray Dataset or DataArray containing the ensemble timeseries.
        title: String to use as the title of the plot.
        era: Optional xarray DataArray containing reference data to plot. Defaults to None.
        time_of_day: List of integers representing hours to filter the valid times. Defaults to [0, 6, 12, 18].
        save: Optional string for the filename. If provided, the plot is saved as a PDF. Defaults to None.
        hline: Optional float representing a specific threshold to plot as a horizontal dashed line. Defaults to None.

    Returns:
        None. Displays a matplotlib figure and optionally saves it to disk.
    """
    tss = tss.isel(step=tss.valid_time.dt.hour.isin(time_of_day).compute())
    times = tss.valid_time.values

    max_temps = tss.max(dim="step")
    top = max_temps.sortby(max_temps, ascending=False).head(number=3)
    top_numbers = top.number.values
    shade = ["#7CFC00", "#32CD32", "#006400"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(
        figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
    )

    p_min = tss.min(dim="number")
    p_max = tss.max(dim="number")
    ax.fill_between(
        times, p_min, p_max, color="grey", alpha=0.3, label="Ensemble Spread"
    )

    for n in tss.number.values:
        ts = tss.sel(number=n)
        # ax.plot(times, ts, linewidth=0.5, color="black", alpha=0.3)
        if n in top_numbers:
            rank = list(top_numbers).index(n)
            ax.plot(times, ts, linewidth=2, color=shade[rank], zorder=5)
        else:
            ax.plot(times, ts, linewidth=0.5, color="black", alpha=0.3)
    ax.plot(
        times,
        tss.mean("number"),
        color="blue",
        linewidth=2,
        linestyle="--",
        label="Ensemble Mean",
    )
    if era is not None:
        era = era.sel(valid_time=times)
        ax.plot(times, era, label="ERA5", linewidth=2, color="red")
    if hline is not None:
        ax.axhline(y=hline, color="red", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(f"Temperature")
    ax.set_xlabel("Days since start")
    ax.legend()
    ax.tick_params(axis="both", which="major")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    if save is not None:
        plt.savefig(f"chaos_plots/{save}.pdf", bbox_inches="tight")
    plt.show()


def fix_lat_lon(data):
    """Tidy latitude and longitude data.

    The function fixes issues from descending values and 0-360 range.

    Args:
        data: A dataset

    Returns:
        data: The fixed dataset.
    """
    data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby(
        "longitude"
    )
    data = data.assign_coords(latitude=data.latitude).sortby("latitude")
    return data


def find_linear_part(x, y):
    """Find the linear part of a curve using the knee/elbow point detection method.

    Normalises both axes to [0, 1] and identifies the knee point as the location
    of maximum deviation from the diagonal (y = x line). All points up to and
    including the knee are considered the linear region.

    Args:
        x: 1D array-like of x values.
        y: 1D array-like of y values, assumed to follow a curve with an initial
           linear region that transitions to a flat curve.

    Returns:
        linear_indices: Boolean array of the same length as x, where True indicates
                        the point that belong to the linear part of the curve.
    """
    # easier with normalised data
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # distance to the y=x line
    distances = y_norm - x_norm

    # point we want is where this distance is the biggest
    knee_idx = np.argmax(distances)

    # mask
    linear_indices = np.arange(len(x)) <= knee_idx

    return linear_indices


class EnsembleChaos:
    """A suite of tools for analyzing the chaotic spread and predictability of ensemble forecasts.

    Provides methods to compare a control run against a perturbed ensemble, calculate 
    Lyapunov exponents, estimate error growth rates, and visualize the spatial 
    and temporal evolution of the forecast spread.

    Args:
        control: xarray Dataset representing the deterministic or control run.
        perturbed: xarray Dataset representing the perturbed ensemble members.
        var_name: String specifying the meteorological variable to analyze. Defaults to "t2m".
    """
    def __init__(self, control, perturbed, var_name="t2m"):
        self.control = fix_lat_lon(control).sortby("step")
        self.perturbed = fix_lat_lon(perturbed).sortby("step")
        self.var_name = var_name
        self.offset = 273.15 if var_name in ["t2m", "tas", "ts", "t2", "2t"] else 0.0

    def plot_trajectories_on_single_point(self, lat, lon, times=[0, 6, 12, 18]):
        """Plot the trajectories for a control run and perturbed ensemble at a single grid point.

        Args:
            lat: Latitude of the point of interest. The nearest grid point will be selected.
            lon: Longitude of the point of interest. The nearest grid point will be selected.
            times: List of hours to filter the data by, if needed. Defaults to [0,6,12,18].

        Returns:
            None. Displays a matplotlib figure showing the ensemble spread (shaded area),
            individual ensemble members (orange lines), and the control run (red line).
        """
        time_mask_control = self.control.valid_time.dt.hour.isin(times)
        time_mask_perturbed = self.perturbed.valid_time.dt.hour.isin(times)

        temp_control = self.control.sel(
            latitude=lat, longitude=lon, method="nearest"
        ).sel(step=time_mask_control)
        temp_perturbed = self.perturbed.sel(
            latitude=lat, longitude=lon, method="nearest"
        ).sel(step=time_mask_perturbed)

        temp_control[self.var_name] = temp_control[self.var_name] - self.offset
        temp_perturbed[self.var_name] = temp_perturbed[self.var_name] - self.offset

        low_bounds = temp_perturbed[self.var_name].min(dim="number")
        up_bounds = temp_perturbed[self.var_name].max(dim="number")

        times = temp_control.valid_time.values

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
        )
        # fill between max and min of ens members
        ax.fill_between(
            times,
            low_bounds,
            up_bounds,
            alpha=0.2,
            color="lightsteelblue",
            label="Ensemble Spread",
        )
        # plot ens members
        for i, n in enumerate(temp_perturbed.number):
            label = "Ensemble Members" if i == 0 else None
            ax.plot(
                times,
                temp_perturbed[self.var_name].sel(number=n),
                alpha=0.3,
                color="cornflowerblue",
                linewidth=1,
                zorder=1,  # in the background
                label=label,
            )
        # plot ens mean
        ens_mean = temp_perturbed[self.var_name].mean(dim="number")
        ax.plot(
            times,
            ens_mean,
            "-.",
            color="midnightblue",
            linewidth=1.5,
            zorder=2,
            label="Ensemble Mean",
        )
        # plot control
        ax.plot(
            times,
            temp_control[self.var_name],
            "D-",
            color="darkorange",
            linewidth=1,
            markersize=2,
            zorder=3,
            label="Control Run",
        )
        ax.set_title(f"Trajectories of {self.var_name.upper()} at lat:{lat} lon:{lon}")
        ax.set_ylabel(f"{self.var_name.upper()}")
        ax.set_xlabel("Day")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_trajectories_on_area_mean(self, lat, lon, times=[0, 6, 12, 18]):
        """Plot the trajectories for a control run and perturbed ensemble averaged over a region.

        Args:
            lat: Slice or array of latitudes defining the region to average over.
            lon: Slice or array of longitudes defining the region to average over.
            times: List of hours to filter the data by, if needed. Defaults to [0,6,12,18].

        Returns:
            None. Displays a matplotlib figure showing the ensemble spread (shaded area),
            individual ensemble members (orange lines), and the control run (red line),
            all spatially averaged over the specified region.
        """
        time_mask_control = self.control.valid_time.dt.hour.isin(times)
        time_mask_perturbed = self.perturbed.valid_time.dt.hour.isin(times)

        weights = np.cos(np.deg2rad(self.control.latitude.sel(latitude=lat)))
        weights.name = "weights"

        temp_control = (
            self.control.sel(latitude=lat, longitude=lon, step=time_mask_control)
            .weighted(weights)
            .mean(dim=["latitude", "longitude"])
        )
        temp_perturbed = (
            self.perturbed.sel(latitude=lat, longitude=lon, step=time_mask_perturbed)
            .weighted(weights)
            .mean(dim=["latitude", "longitude"])
        )

        temp_control[self.var_name] = temp_control[self.var_name] - self.offset
        temp_perturbed[self.var_name] = temp_perturbed[self.var_name] - self.offset

        low_bounds = temp_perturbed[self.var_name].min(dim="number")
        up_bounds = temp_perturbed[self.var_name].max(dim="number")

        times = temp_control.valid_time.values

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
        )

        # fill between max and min of ens members
        ax.fill_between(
            times,
            low_bounds,
            up_bounds,
            alpha=0.2,
            color="lightsteelblue",
            label="Ensemble Spread",
        )

        # plot ens members
        for i, n in enumerate(temp_perturbed.number):
            label = "Ensemble Members" if i == 0 else None
            ax.plot(
                times,
                temp_perturbed[self.var_name].sel(number=n),
                alpha=0.3,
                color="cornflowerblue",
                linewidth=1,
                zorder=1,  # in the background
                label=label,
            )

        # plot ens mean
        ens_mean = temp_perturbed[self.var_name].mean(dim="number")
        ax.plot(
            times,
            ens_mean,
            "-.",
            color="midnightblue",
            linewidth=1.5,
            zorder=2,
            label="Ensemble Mean",
        )

        # plot control
        ax.plot(
            times,
            temp_control[self.var_name],
            "D-",
            color="darkorange",
            linewidth=1,
            markersize=2,
            zorder=3,
            label="Control Run",
        )

        ax.set_title(f"Area Mean Trajectories of {self.var_name.upper()}")
        ax.set_ylabel(f"{self.var_name.upper()}")
        ax.set_xlabel("Day")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def lyapunov_single_point(self, lat, lon, times=[0, 6, 12, 18]):
        """Estimate the Lyapunov exponent at a single grid point.

        Computes the log-RMSE between each perturbed ensemble member and the control run at the nearest grid point, then fits a line to the initial linear growth region to estimate the Lyapunov exponent.

        Args:
            lat: Latitude of the point of interest. The nearest grid point will be selected.
            lon: Longitude of the point of interest. The nearest grid point will be selected.
            times: List of hours to filter the data by. Defaults to [0,6,12,18].

        Returns:
            None. Displays a matplotlib figure showing individual log-RMSE trajectories
            (blue), the ensemble mean (orange), and the linear fit (green dashed), and
            prints the estimated Lyapunov exponent in days^-1.
        """
        # mask to only compute on selected times for each day
        time_mask_control = self.control.valid_time.dt.hour.isin(times)
        time_mask_perturbed = self.perturbed.valid_time.dt.hour.isin(times)

        # select only the data on the point we're interested in and times we want
        temp_control = self.control.sel(
            latitude=lat, longitude=lon, method="nearest"
        ).sel(step=time_mask_control)
        temp_perturbed = self.perturbed.sel(
            latitude=lat, longitude=lon, method="nearest"
        ).sel(step=time_mask_perturbed)

        # extract times for better plotting
        time_indexes = temp_control.valid_time.values
        time_since_start = (time_indexes - time_indexes[0]).astype(
            "timedelta64[h]"
        ).astype(float) / 24.0

        # computation of logdist
        sq_diff = (temp_control[self.var_name] - temp_perturbed[self.var_name]) ** 2
        rmse = np.sqrt(sq_diff)
        logdist = np.log(rmse)
        mean_logdist = logdist.mean(dim="number")
        mean_logdist_np = mean_logdist.values

        # compute the lyapunov exponent

        linear_indices = find_linear_part(time_since_start, mean_logdist_np)
        slope, intercept = np.polyfit(
            time_since_start[linear_indices], mean_logdist_np[linear_indices], 1
        )
        lyapunov_exponent = np.round(slope, 3)

        # plot and print
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
        )
        ax.set_ylabel("$log(RMSE)$")
        ax.set_xlabel("Day")

        for i, n in enumerate(logdist.number):
            label = "Ensemble Members" if i == 0 else None
            ax.plot(
                time_since_start,
                logdist.sel(number=n),
                color="cornflowerblue",
                alpha=0.1,
                zorder=1,
                label=label,
            )

        ax.plot(
            time_since_start,
            mean_logdist,
            "o-",
            color="midnightblue",
            linewidth=1.5,
            zorder=2,
            label="Mean $log(RMSE)$",
        )

        fit_y = slope * time_since_start[linear_indices] + intercept
        ax.plot(
            time_since_start[linear_indices],
            fit_y,
            color="darkorange",
            linewidth=2,
            ls="--",
            zorder=3,
            label=f"Linear Part (λ = {lyapunov_exponent})",
        )

        ax.set_title(f"Lyapunov Exponent at lat:{lat} lon:{lon}")
        ax.legend()
        plt.tight_layout()
        plt.show()
        print(f"Estimated Lyapunov Exponent (λ): {lyapunov_exponent} days^-1")

    def growth_rate_pairwise_bootstrap(
        self, lat, lon, times=[0, 6, 12, 18], N=100, save=None, crps=False
    ):
        """Estimate the upper-bound predictability limit and growth rate (alpha) over a spatial region.

        Unlike lyapunov_over_area_pairwise which uses a linear fit on log-RMSE,
        this function computes the root-mean-square error (RMSE) for all unique pairs
        and fits the Lorenz logistic growth model (dE/dt = alpha * E * (1 - E/E_inf))
        to estimate the error growth rate (alpha) and the saturation error (E_inf).

        Args:
            lat: Slice or array of latitudes defining the region to average over.
            lon: Slice or array of longitudes defining the region to average over.
            times: List of hours to filter the data by. Defaults to [0,6,12,18].

        Returns:
            alpha, E_inf: Estimated growth rate (days^-1) and saturation error, rounded to 3
                          decimal places. Also displays a matplotlib figure and prints the result.
        """
        time_mask_control = self.control.valid_time.dt.hour.isin(times)
        time_mask_perturbed = self.perturbed.valid_time.dt.hour.isin(times)

        # the first input is to same if using crps
        if crps:
            time_mask_control[0] = False
            time_mask_perturbed[0] = False

        # select only the area on the point we're interested in and times we want
        temp_control = self.control.sel(latitude=lat, longitude=lon).sel(
            step=time_mask_control
        )[self.var_name]
        temp_perturbed = self.perturbed.sel(latitude=lat, longitude=lon).sel(
            step=time_mask_perturbed
        )[self.var_name]

        # computation of logdist
        data = (
            xr.concat(
                [temp_control.expand_dims("number"), temp_perturbed], dim="number"
            )
            .stack(space=["latitude", "longitude"])
            .transpose("step", "number", "space")
        )

        # Subtract the ensemble mean to center the data around 0 (trying to avoid catastrophic annulation)
        data = data - data.mean(dim="number", skipna=True)

        # weights
        weights = np.cos(np.deg2rad(data.latitude.values))
        weights_norm = weights / np.mean(weights)

        # compute RMSE pairwise relatively fast
        v = data.values  # extract numpy array
        sq_mean = np.mean(
            (v**2) * weights_norm, axis=-1
        )  # compute the squared mean of the data for an ensemble and step [step x number]
        # dark magic for the compute of the cross term when expanding the square, first apply the weights on one of the terms then
        # perform a batched matrix multiplication (@) to calculate the dot product of every ensemble pair (2AB) acrros all time steps
        # the final .transpose(1, 2, 0) transposes the result from (step, number, number) to (number, number, step) so that tri_indices works fine
        rmse = np.sqrt(
            np.maximum(
                (
                    sq_mean[:, :, None]
                    + sq_mean[:, None, :]
                    - 2 * ((v * weights_norm) @ v.transpose(0, 2, 1)) / v.shape[-1]
                ).transpose(1, 2, 0),
                0,
            )
        )
        # max in case matmul gives smth <0, happens sometimes when numbers are very small

        # Pairwise extractions and then mean over all ensemble pairs
        pairwise_rmse = rmse[np.triu_indices(rmse.shape[0], k=1)]
        logdist = np.log(pairwise_rmse)
        mean_rmse = np.mean(pairwise_rmse, axis=0)
        mean_logdist = np.mean(logdist, axis=0)

        # extract times for better plotting
        time_indexes = temp_control.valid_time.values
        time_since_start = (time_indexes - time_indexes[0]).astype(
            "timedelta64[h]"
        ).astype(float) / 24.0

        if crps:
            time_since_start = time_since_start + 0.25

        # fit lorenz model
        # solution of lorenz model for growth
        E0 = mean_rmse[0]

        def logistic_solution(t, alpha_param, E_inf_param):
            return E_inf_param / (
                1.0 + ((E_inf_param - E0) / E0) * np.exp(-alpha_param * t)
            )

        max_E = np.max(mean_rmse)

        popt, _ = curve_fit(
            logistic_solution,
            time_since_start,
            mean_rmse,
            p0=[0.3, max_E],
            bounds=([0.0, 0.0], [1, max_E * 5.0]),
        )

        alpha = np.round(popt[0], 3)
        E_inf = np.round(popt[1], 3)

        # bootstrap (the first fit was done to warmup the bootstraps)

        num_pairs = pairwise_rmse.shape[0]
        bootstrap_alphas, bootstrap_E_infs = [], []

        for n in range(N):
            idx = np.random.choice(num_pairs, num_pairs, replace=True)
            bootstrap_mean_rmse = np.mean(pairwise_rmse[idx], axis=0)

            popt_bootstrap, _ = curve_fit(
                logistic_solution,
                time_since_start,
                bootstrap_mean_rmse,
                p0=[alpha, E_inf],
                bounds=([0.0, 0.0], [1, max_E * 5.0]),
                maxfev=2000,
            )
            bootstrap_alphas.append(popt_bootstrap[0])
            bootstrap_E_infs.append(popt_bootstrap[1])

        alpha_ci_plusminus = 1.96 * np.std(bootstrap_alphas)
        alpha_lower = np.round(alpha - alpha_ci_plusminus, 3)
        alpha_upper = np.round(alpha + alpha_ci_plusminus, 3)

        E_ci_plusminus = 1.96 * np.std(bootstrap_E_infs)
        E_inf_lower = np.round(E_inf - E_ci_plusminus, 3)
        E_inf_upper = np.round(E_inf + E_ci_plusminus, 3)

        # plot and print
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
        )
        ax.set_ylabel("$log(RMSE)$")
        ax.set_xlabel("Day")

        for n in range(logdist.shape[0]):
            label = "Pairwise Differences" if n == 0 else None
            ax.plot(
                time_since_start,
                logdist[n, :],
                color="grey",
                alpha=0.5,
                linewidth=1,
                zorder=1,
                label=label,
            )

        ax.plot(
            time_since_start,
            mean_logdist,
            "o-",
            color="black",
            linewidth=2,
            zorder=2,
            label="Mean $log(RMSE)$",
        )

        # fit for plotting
        E_theoretical = logistic_solution(time_since_start, alpha, E_inf)
        log_E_theoretical = np.log(E_theoretical)

        ax.plot(
            time_since_start,
            log_E_theoretical,
            color="blue",
            linewidth=2,
            ls="--",
            zorder=3,
            label=f"Fit (α $\in$ [{alpha_lower},{alpha_upper}])",
        )

        # ax.set_title("Log Differences (Pairwise)",)
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"lyap_plots/{save}.png", dpi=300)
        plt.show()

        print(
            f"Estimated Growth Rate (α): {alpha} [95% CI: {alpha_lower}, {alpha_upper}] days^-1 | "
            f"Saturation Error (E_inf): {E_inf} [95% CI: {E_inf_lower}, {E_inf_upper}]"
        )

        # return alpha, E_inf, (alpha_lower, alpha_upper), (E_inf_lower, E_inf_upper)

    def lyapunov_over_area_pairwise_bootstrap(
        self, lat, lon, times=[0, 6, 12, 18], N=100, save=None, crps=False
    ):
        """Estimate the Lyapunov exponent using all pairwise differences over a spatial region.

        Unlike lyapunov_over_area, which compares each ensemble member to the control,
        this function computes log-RMSE for all unique pairs across the combined set of
        control and perturbed members (including control vs. each member and member vs.
        member). The Lyapunov exponent is then estimated from the linear growth region
        of the mean pairwise log-RMSE.

        Args:
            lat: Slice or array of latitudes defining the region to average over.
            lon: Slice or array of longitudes defining the region to average over.
            times: List of hours to filter the data by. Defaults to [0,6,12,18].

        Returns:
            lyapunov_exponent: Estimated Lyapunov exponent in days^-1, rounded to 3
                               decimal places. Also displays a matplotlib figure and
                               prints the result.
        """
        time_mask_control = self.control.valid_time.dt.hour.isin(times)
        time_mask_perturbed = self.perturbed.valid_time.dt.hour.isin(times)

        # the first input is to same if using crps
        if crps:
            time_mask_control[0] = False
            time_mask_perturbed[0] = False

        # select only the area on the point we're interested in and times we want
        temp_control = self.control.sel(latitude=lat, longitude=lon).sel(
            step=time_mask_control
        )[self.var_name]
        temp_perturbed = self.perturbed.sel(latitude=lat, longitude=lon).sel(
            step=time_mask_perturbed
        )[self.var_name]

        # computation of logdist
        data = (
            xr.concat(
                [temp_control.expand_dims("number"), temp_perturbed], dim="number"
            )
            .stack(space=["latitude", "longitude"])
            .transpose("step", "number", "space")
        )

        # Subtract the ensemble mean to center the data around 0 (trying to avoid catastrophic annulation)
        data = data - data.mean(dim="number", skipna=True)

        # weights
        weights = np.cos(np.deg2rad(data.latitude.values))
        weights_norm = weights / np.mean(weights)

        # compute RMSE pairwise relatively fast
        v = data.values  # extract numpy array
        sq_mean = np.mean(
            (v**2) * weights_norm, axis=-1
        )  # compute the squared mean of the data for an ensemble and step [step x number]
        # dark magic for the compute of the cross term when expanding the square, first apply the weights on one of the terms then
        # perform a batched matrix multiplication (@) to calculate the dot product of every ensemble pair (2AB) acrros all time steps
        # the final .transpose(1, 2, 0) transposes the result from (step, number, number) to (number, number, step) so that tri_indices works fine
        rmse = np.sqrt(
            np.maximum(
                (
                    sq_mean[:, :, None]
                    + sq_mean[:, None, :]
                    - 2 * ((v * weights_norm) @ v.transpose(0, 2, 1)) / v.shape[-1]
                ).transpose(1, 2, 0),
                0,
            )
        )

        # Pairwise extractions and then mean over all ensemble pairs
        pairwise_rmse = rmse[np.triu_indices(rmse.shape[0], k=1)]
        logdist = np.log(pairwise_rmse)
        mean_rmse = np.mean(pairwise_rmse, axis=0)
        mean_logdist = np.mean(logdist, axis=0)

        # extract times for better plotting
        time_indexes = temp_control.valid_time.values
        time_since_start = (time_indexes - time_indexes[0]).astype(
            "timedelta64[h]"
        ).astype(float) / 24.0

        if crps:
            time_since_start = time_since_start + 0.25

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.5)
        )
        ax.set_ylabel("$log(RMSE)$")
        ax.set_xlabel("Day")

        for n in range(logdist.shape[0]):
            label = "Pairwise Differences" if n == 0 else None
            ax.plot(
                time_since_start,
                logdist[n, :],
                color="cornflowerblue",
                alpha=0.05,
                zorder=1,
                label=label,
            )

        ax.plot(
            time_since_start,
            mean_logdist,
            "o-",
            color="midnightblue",
            linewidth=1.5,
            zorder=2,
            label="Mean $log(RMSE)$",
        )

        # compute the lyapunov exponent

        linear_indices = find_linear_part(time_since_start, mean_logdist)
        slope, intercept = np.polyfit(
            time_since_start[linear_indices], mean_logdist[linear_indices], 1
        )
        lyapunov_exponent = np.round(slope, 3)

        # bootstrap

        num_pairs = logdist.shape[0]
        bootstrap_lyaps = []

        for n in range(N):
            idx = np.random.choice(num_pairs, num_pairs, replace=True)
            bootstrap_mean_logdist = np.mean(logdist[idx], axis=0)

            slope_boot, _ = np.polyfit(
                time_since_start[linear_indices],
                bootstrap_mean_logdist[linear_indices],
                1,
            )
            bootstrap_lyaps.append(slope_boot)

        lyap_ci_plusminus = 1.96 * np.std(bootstrap_lyaps)
        lyap_lower = np.round(lyapunov_exponent - lyap_ci_plusminus, 3)
        lyap_upper = np.round(lyapunov_exponent + lyap_ci_plusminus, 3)

        # plot and print

        fit_y = slope * time_since_start[linear_indices] + intercept
        ax.plot(
            time_since_start[linear_indices],
            fit_y,
            color="darkorange",
            linewidth=2,
            ls="--",
            zorder=3,
            label=f"Linear Part (λ $\in$ [{lyap_lower},{lyap_upper}])",
        )

        ax.set_title("Area Mean Lyapunov Exponent (Pairwise)")
        ax.legend()
        plt.tight_layout()
        if save is not None:
            plt.savefig(f"lyap_plots/{save}_lyapunov_basic.png", dpi=300)
        plt.show()

        print(f"Estimated Lyapunov Exponent (λ): {lyapunov_exponent} days^-1")

    #        return lyapunov_exponent

    def plot_nice_looking_animation(
        self, lat_bnds, lon_bnds, member=0, filename=None, speed=300, cmap="viridis"
    ):
        """Animate a temperature field over time using a PlateCarree projection.

        Converts temperature from Kelvin to Celsius and renders an animated
        pcolormesh map over the specified region, with a consistent colorscale
        across all frames.

        Args:
            lat_bnds: slice defining the latitude bounds of the region to display
                      and use for colorscale normalisation (e.g. slice(40, 60)).
            lon_bnds: slice defining the longitude bounds of the region to display
                      and use for colorscale normalisation (e.g. slice(-10, 20)).
            member: ensemble member to choose (0=control)
            filename: optional filename for saving the figure

        Returns:
            HTML or None: Jupyter-renderable HTML object or None if saved to file.
        """
        if member == 0:
            data = self.control[self.var_name]
            member_label = "Control Run"
        else:
            data = self.perturbed[self.var_name].sel(number=member)
            member_label = f"Perturbed Member {member}"

        data = data - self.offset

        vmin, vmax = (
            data.sel(latitude=lat_bnds, longitude=lon_bnds).min().item(),
            data.sel(latitude=lat_bnds, longitude=lon_bnds).max().item(),
        )

        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        fig.tight_layout()

        mesh = data.isel(step=0).plot.pcolormesh(
            ax=ax, levels=30, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False
        )

        ax.set_extent(
            [lon_bnds.start, lon_bnds.stop, lat_bnds.start, lat_bnds.stop],
            crs=ccrs.PlateCarree(),
        )
        ax.coastlines()

        # Update loop
        def update(frame):
            step_data = data.isel(step=frame)
            mesh.set_array(step_data.values.ravel())
            time_str = np.datetime_as_string(step_data.valid_time.values, unit="h")
            ax.set_title(f"[{member_label}] Step: {frame} | Time: {time_str}")

        ani = animation.FuncAnimation(
            fig, update, frames=len(data.step), interval=speed
        )

        if filename:
            ani.save(filename, writer="pillow", dpi=300)
            plt.close(fig)
            return None
        else:
            plt.close(fig)
            return HTML(ani.to_jshtml())

    def plot_nice_looking_animation_ortho(
        self, lat_bnds, lon_bnds, member=0, filename=None
    ):
        """Animate a temperature field over time using an Orthographic projection.

        Similar to plot_nice_looking_animation, but renders the data on a globe
        centred on the middle of the specified region, providing a more
        three-dimensional perspective of the spatial data.

        Args:
            lat_bnds: slice defining the latitude bounds used for colorscale
                      normalisation and to compute the projection centre latitude
                      (e.g. slice(40, 60)).
            lon_bnds: slice defining the longitude bounds used for colorscale
                      normalisation and to compute the projection centre longitude
                      (e.g. slice(-10, 20)).
            member: ensemble member to choose (0=control)
            filename: optional filename for saving the figure

        Returns:
            HTML or None: Jupyter-renderable HTML object or None if saved to file.
        """
        if member == 0:
            data = self.control[self.var_name]
            member_label = "Control Run"
        else:
            data = self.perturbed[self.var_name].sel(number=member)
            member_label = f"Perturbed Member {member}"

        data = data - self.offset

        vmin, vmax = (
            data.sel(latitude=lat_bnds, longitude=lon_bnds).min().item(),
            data.sel(latitude=lat_bnds, longitude=lon_bnds).max().item(),
        )

        center_lon = (lon_bnds.start + lon_bnds.stop) / 2
        center_lat = (lat_bnds.start + lat_bnds.stop) / 2

        fig, ax = plt.subplots(
            figsize=get_figsize(WIDTH_INSA, fraction=1.0, height_factor=0.8),
            subplot_kw={"projection": ccrs.Orthographic(center_lon, center_lat)},
        )

        mesh = data.isel(step=0).plot.pcolormesh(
            ax=ax,
            levels=30,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines()
        padding = 60
        ax.set_extent(
            [
                lon_bnds.start - padding,
                lon_bnds.stop + padding,
                lat_bnds.start - padding,
                lat_bnds.stop + padding,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Update loop
        def update(frame):
            step_data = data.isel(step=frame)
            mesh.set_array(step_data.values.ravel())
            time_str = np.datetime_as_string(step_data.valid_time.values, unit="h")
            ax.set_title(f"[{member_label}] Step: {frame} | Time: {time_str}")

        ani = animation.FuncAnimation(fig, update, frames=len(data.step), interval=300)

        if filename:
            ani.save(filename, writer="pillow")
            plt.close(fig)
            return None
        else:
            plt.close(fig)
            return HTML(ani.to_jshtml())
