import numpy as np


def sph2cart_2d(r, az):
    # Radar convention: az=0 is forward (Y axis), positive az is right (X axis)
    # x = r*sin(az),  y = r*cos(az)
    return np.array([r * np.sin(az), r * np.cos(az)])

def cart2sph_2d(x, y):
    # Inverse: az = atan2(x, y)
    r = np.hypot(x, y)
    az = np.arctan2(x, y)
    return r, az

def calc_gating_limits_2d(P, H, R=None):
    """
    Calculate the gating limits for a 2D Kalman filter.

    Parameters
    ----------
    P : np.ndarray
        The state covariance matrix.
    H : np.ndarray
        The measurement matrix.
    R : np.ndarray, optional
        The measurement noise covariance matrix. If None, an error is raised.

    Returns
    -------
    S : np.ndarray
        The innovation covariance matrix.
    S_inv : np.ndarray
        The inverse of the innovation covariance matrix.
    """

    if R is None:
        raise ValueError("R must be provided")
    S = H @ P @ H.T + R
    return S, np.linalg.inv(S)

def compute_mahalanobis_2d(residual, S_inv):
    """
    Compute the Mahalanobis distance for a 2D residual vector.

    Parameters
    ----------
    residual : np.ndarray
        The residual vector (difference between predicted and observed values).
    S_inv : np.ndarray
        The inverse of the innovation covariance matrix.

    Returns
    -------
    float
        The Mahalanobis distance.
    """

    return float(residual.T @ S_inv @ residual)

def wrap_angle(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Parameters
    ----------
    angle : float or np.ndarray
        The angle in radians to be wrapped.

    Returns
    -------
    float or np.ndarray
        The wrapped angle in radians, within the range [-pi, pi].
    """

    return (angle + np.pi) % (2 * np.pi) - np.pi