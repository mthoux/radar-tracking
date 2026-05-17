import numpy as np



def hardcoded_get_ant_static_2d():
    """
    Computes virtual antenna positions for static radar setup.

    Returns:
    -------
        x_ant_pos: np.ndarray of virtual antenna x-positions
        z_ant_pos: np.ndarray of virtual antenna z-positions
    """

    lm = 3e8 / 77e9  # lambda for 77 GHz

    RX_X = np.array([-3*lm/2, -lm, -lm/2, 0])      # 4 Rx
    RX_Z = np.array([0, 0, 0, 0])
    TX_X = np.array([0, lm, 2*lm])                 # 3 Tx
    TX_Z = np.array([0, lm/2, 0])

    x_ant_pos = []
    z_ant_pos = []

    for tx_i in range(len(TX_X)):
        for rx_i in range(len(RX_X)):
            x = TX_X[tx_i] + RX_X[rx_i]
            z = TX_Z[tx_i] + RX_Z[rx_i]
            x_ant_pos.append(x)
            z_ant_pos.append(z)

    x_ant_pos = np.array(x_ant_pos)
    z_ant_pos = np.array(z_ant_pos)

    # Optional: make origin zero-centered
    x_ant_pos -= np.min(x_ant_pos)
    z_ant_pos -= np.min(z_ant_pos)

    return x_ant_pos, z_ant_pos

# Helper function to get point cloud values
def plot_3d_cart_heatmap(ax,voxel,xaxis,yaxis,zaxis,threshold):
    ''''
    Returns X,Y,Z positions of voxels with power above a threshold.

    Parameters:
    - ax: matplotlib 3D axis to plot on
    - xaxis: x-values (for BF azimuth angles, for MF x distances)
    - yaxis: y-values (for BF elevation angles, for MF y distances)
    - zaxis: z-values (for range bins, for MF z distances)
    
    Returns:
    - X_: x points
    - Y_: y points
    - Z_: z points
    - intesn: intensity of the points (used for coloring)
    '''

    thresh = np.max(np.abs(voxel)) * threshold
    
    # Find indices where voxel values exceed the threshold
    ptcloud_lim = thresh
    pc_idx = np.where(voxel > ptcloud_lim)
    print(len(pc_idx))

    # Convert indices to subscripts
    x_idx, y_idx, z_idx = pc_idx[0],pc_idx[1],pc_idx[2]
    # Extract corresponding coordinates
    X_ = xaxis[x_idx]
    Y_ = yaxis[y_idx]
    Z_ = zaxis[z_idx]
    intesn = voxel[x_idx, y_idx, z_idx] 

    ax.scatter(X_,Y_, Z_, c=intesn, cmap='jet', marker='o')
    ax.view_init(elev=45, azim=45)  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add a grid and make it interactive (movable)
    ax.grid(True)

    # return X_, Y_, Z_, intesn 


def load_raw_data(data_path):
    """
    Load raw radar data from a .mat file.

    Parameters:
    ----------
    data_path : str
        Path to the .mat file containing the raw radar data.

    Returns:
    -------
    radar_params : dict
        Dictionary containing radar parameters such as sample rate, number of samples, etc.
    raw_data : np.ndarray
        The raw radar data reshaped to (num_x_stp, num_z_stp, adc_samples).
    """

    import scipy.io as sio
    mat_data = sio.loadmat(data_path)

    raw_data = mat_data['data_raw']  # shape: (frames, tx, rx, samples)

    num_frames, num_tx, num_rx, adc_samples = raw_data.shape
    num_x_stp = num_tx * num_rx
    num_z_stp = num_frames

    # Reshape to (num_x_stp, num_z_stp, adc_samples)
    raw_data = raw_data.transpose(1, 2, 0, 3)  # (tx, rx, frames, samples)
    raw_data = raw_data.reshape(num_tx * num_rx, num_frames, adc_samples)

    radar_params = {
        'sample_rate': 10e6,
        'num_samples': adc_samples,
        'slope': 70.150e12,
        'lm': 3e8 / 77e9,
        'num_x_stp': num_x_stp,
        'num_z_stp': num_z_stp,
        'num_tx': num_tx,
        'num_rx': num_rx,
        'adc_samples': adc_samples,
        'num_frames': num_frames
    }

    return radar_params, raw_data


def sph2cart(az, el, r):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    ----------
    az : array_like
        Azimuthal angle in radians.
    el : array_like
        Polar angle in radians.
    r : array_like
        Radius (distance from the origin).

    Returns:
    ----------
    x : array_like
        x-coordinate in Cartesian coordinates.
    y : array_like
        y-coordinate in Cartesian coordinates.
    z : array_like
        z-coordinate in Cartesian coordinates.
    """

    y = r * np.sin(el)
    rcosel = r * np.cos(el)
    x = rcosel * np.cos(az)
    z = rcosel * np.sin(az)
    return x, y, z


# Function to plot a 2D heatmap in polar coordinates
def plot_2d_heatmap(ax, data, theta, r, vmin=0, vmax=0.1):
    """
    Plot a 2D heatmap in polar coordinates.

    Parameters:
    ----------
        data: 2D numpy array
            The heatmap data to be plotted. Of size (theta x r)
        r_max: float
            Maximum radius of the polar plot.
    """

    R, Theta  = np.meshgrid(r,theta)

    ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlim(theta[0],theta[-1])
    ax.set_ylim(r[0],r[-1])
    ax.grid(False)

# Function to plot a 2D heatmap in polar coordinates
def plot_2d_polar_heatmap(ax, data, az, el, vmin=0, vmax=0.1):
    """
    Plot a 2D heatmap in polar coordinates.

    Parameters:
    ----------
        data: 2D numpy array
            The heatmap data to be plotted.
        r_max: float
            Maximum radius of the polar plot.
    """

    # Create the heatmap
    ax.pcolormesh(az, el, data.T, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)

    # Label axes
    ax.set_xlabel(r"$\theta$ (Azimuthal Angle, radians)")
    ax.set_ylabel(r"$\phi$ (Polar Angle, radians)")
    ax.grid(False)
    ax.title.set_text("2D Polar Heatmap (φ-θ)")

# Function to plot a 3D polar heatmap as a point cloud
def plot_3d_polar_heatmap(ax, data, az, el,r,threshold):
    """
    Plot a 3D heatmap in spherical coordinates as a point cloud.

    Parameters:
    ----------
        data: 3D numpy array
            The heatmap data to be plotted. Should have shape (n_r, n_phi, n_theta).
        r_max: float
            Maximum radius of the spherical coordinates.
    """

    # Create a meshgrid of spherical coordinates
    R, Phi, Theta = np.meshgrid(r, az, el, indexing='ij')

    # Convert spherical coordinates to Cartesian for plotting
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)

    # Flatten arrays for point cloud
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    values = data.flatten()
    thresh = np.max(np.abs(values)) * threshold
    
    # Find indices where voxel values exceed the threshold
    ptcloud_lim = thresh
    pc_idx = np.where(values > ptcloud_lim)

    # Convert indices to subscripts
    idx = pc_idx[0]
    # Extract corresponding coordinates
    X_ = x[idx]
    Y_ = y[idx]
    Z_ = z[idx]
    intesn = idx 

    # Plot the point cloud
    ax.scatter(X_, Y_, Z_, c=intesn, cmap='jet', s=10)
    ax.title.set_text("3D Polar Heatmap Point Cloud")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def cart2pol(x_flat, y_flat):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x_flat : np.ndarray
        Array of x-coordinates.
    y_flat : np.ndarray
        Array of y-coordinates.

    Returns
    -------
    np.ndarray
        Array of polar coordinates in the form of (phi, r), where:
        - phi is the azimuthal angle in radians.
        - r is the radial distance from the origin.
    """

    phi_flat = np.arctan2(y_flat, x_flat)
    r_flat = np.hypot(x_flat, y_flat)

    return np.column_stack((phi_flat, r_flat))



