import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal(ax):
    """
    Set 3D axes to equal scale so the ellipsoid is not visually distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_ellipsoid_with_axes(
    a=1.4,
    b=1.0,
    c=0.7,
    n=90,
    axis_extend=1.5,
    label_offset=1.06,
    elev=18,
    azim=35,
    save_path=None,
):
    """
    Draw an anisotropic ellipsoid with RGB coordinate axes.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths along X/Y/Z.
    n : int
        Sampling density for the ellipsoid mesh.
    axis_extend : float
        Axes length multiplier relative to (a,b,c). >1 makes axes exceed ellipsoid.
    label_offset : float
        Label position multiplier relative to axis tip. >1 pushes labels outward.
    elev, azim : float
        View angles for matplotlib 3D camera.
    save_path : str or None
        If provided, save figure to this path (e.g., "ellipsoid.png").
    """
    # Parameterization
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)

    x = a * np.sin(vv) * np.cos(uu)
    y = b * np.sin(vv) * np.sin(uu)
    z = c * np.cos(vv)

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # --- Pure white background, no grids ---
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)

    # Remove gridlines/panes artifacts (keep axes)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    # Make panes transparent/white
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))

    # Optional: hide 3D box edge lines for cleaner look
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

    # --- Ellipsoid ---
    ax.plot_surface(
        x, y, z,
        rstride=2, cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.18
    )
    ax.plot_wireframe(
        x, y, z,
        rstride=6, cstride=6,
        linewidth=0.7
    )

    # --- RGB axes that extend beyond ellipsoid ---
    x_color = "#d62728"  # red
    y_color = "#2ca02c"  # green
    z_color = "#1f77b4"  # blue

    ax_len_x = a * axis_extend
    ax_len_y = b * axis_extend
    ax_len_z = c * axis_extend

    ax.plot([0, ax_len_x], [0, 0], [0, 0], linewidth=3, color=x_color)
    ax.plot([0, 0], [0, ax_len_y], [0, 0], linewidth=3, color=y_color)
    ax.plot([0, 0], [0, 0], [0, ax_len_z], linewidth=3, color=z_color)

    # Labels placed slightly beyond axis tips
    ax.text(ax_len_x * label_offset, 0, 0, "X", fontsize=14, color=x_color)
    ax.text(0, ax_len_y * label_offset, 0, "Y", fontsize=14, color=y_color)
    ax.text(0, 0, ax_len_z * label_offset, "Z", fontsize=14, color=z_color)

    # --- Limits and view ---
    lim = 1.20 * max(ax_len_x, ax_len_y, ax_len_z)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("Anisotropic Ellipsoid", pad=10)

    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()


if __name__ == "__main__":
    # Example similar to your screenshot: flatter in Z, longer in X
    plot_ellipsoid_with_axes(
        a=1.4, b=1.0, c=0.7,
        axis_extend=1.22,   # axes slightly longer than ellipsoid
        label_offset=1.05,  # labels slightly beyond axis tips
        elev=18, azim=35,
        save_path=None
    )
