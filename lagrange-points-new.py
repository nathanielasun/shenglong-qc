import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants (SI units)
G = 1.0  # Gravitational constant
M_sun = 1.989e30  # Solar mass
M_jup = 1.898e27  # Jupiter mass
a_jup = 7.783e11  # Jupiter semi-major axis (m)
e_jup = 0.0489  # Jupiter eccentricity
inc_jup = 1.308  # Jupiter inclination
long_asc_jup = 273.96  # Longitude of ascending node
arg_peri_jup = 23.36  # Argument of perihelion
true_long_jup = 111.08 # True longitude (at epoch)
n_jup = np.sqrt(G * M_sun / a_jup**3)  # Mean motion

# Time parameters
T = 200 * 3600 * 24  # Integration time (200 days)
dt = 3600 * 24  # Time step (1 day)

# Grid parameters
N = 50  # Number of points in grid
extent = 1.5e11  # Extend grid in meters (1.5 AU)
xs = np.linspace(a_jup - extent, a_jup + extent, N)
ys = np.linspace(-extent, extent, N)


def jupiter_position_keplerian(t):
    """Calculates Jupiter's position using Kepler's equation and orbital elements."""
    M = n_jup * t
    E = M  # Initial guess for eccentric anomaly
    for _ in range(10):  # Iterative solution of Kepler's equation
        E = M + e_jup * np.sin(E)
    v = 2 * np.arctan2(np.sqrt(1 + e_jup) * np.sin(E / 2), np.sqrt(1 - e_jup) * np.cos(E / 2))

    x_jup = a_jup * (np.cos(E) - e_jup)
    y_jup = a_jup * np.sqrt(1 - e_jup**2) * np.sin(E)

    # Convert to cartesian coordinates in ecliptic plane
    x_ecl = x_jup * np.cos(inc_jup * np.pi / 180) - y_jup * np.sin(inc_jup * np.pi / 180) * np.cos(long_asc_jup * np.pi / 180)
    y_ecl = x_jup * np.sin(inc_jup * np.pi / 180) + y_jup * np.cos(inc_jup * np.pi / 180) * np.cos(long_asc_jup * np.pi / 180)
    
    return np.array([x_ecl, y_ecl])


def acceleration_sun_jupiter(r, t):
    """Calculates acceleration due to Sun and Jupiter."""
    rj = jupiter_position_keplerian(t)
    a_sun = -G * M_sun * r / np.linalg.norm(r)**3
    a_jup = -G * M_jup * (r - rj) / np.linalg.norm(r - rj)**3
    return a_sun + a_jup


def integrate_dynamics(t, y, sun_mass, jupiter_mass, G):
  """
  Defines the system of differential equations for the motion.
  """
  r = y[:2]
  v = y[2:]
  
  a = -G * sun_mass * r / np.linalg.norm(r)**3
  rj = jupiter_position_keplerian(t)
  a += -G * jupiter_mass * (r - rj) / np.linalg.norm(r - rj)**3

  return np.concatenate((v, a))


def ftle_estimate(r0, v0, T, dt, eps=1e-6):
    """Estimates FTLE using solve_ivp."""
    sol1 = solve_ivp(
        lambda t, y: integrate_dynamics(t, y, M_sun, M_jup, G),
        [0, T],
        np.concatenate((r0, v0)),
        dense_output=True,
        max_step=dt,  # Important for accuracy
    )
    
    sol2 = solve_ivp(
        lambda t, y: integrate_dynamics(t, y, M_sun, M_jup, G),
        [0, T],
        np.concatenate((r0 + np.array([eps, 0]), v0)),
        dense_output=True,
        max_step=dt,
    )
    
    r1_final = sol1.y[:2, -1]
    r2_final = sol2.y[:2, -1]
    
    dT = np.linalg.norm(r2_final - r1_final)
    return np.log(dT / eps) / T


# Create FTLE map
ftle_map = np.zeros((N, N))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        r0 = np.array([x, y])
        # Initial velocity:  Circular approximation - refined
        v0 = np.array([0.0, np.sqrt(G * M_sun / (np.linalg.norm(r0))**2)])
        ftle_map[j, i] = ftle_estimate(r0, v0, T, dt)


# Plot FTLE map
plt.imshow(
    ftle_map,
    extent=[xs[0], xs[-1], ys[0], ys[-1]],
    origin='lower',
    cmap='winter',
)
plt.colorbar(label='FTLE')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Lagrangian Stability Map near Jupiter (Improved)')
plt.show()

