import numpy as np

G = 1.0
M_sun = 1.0
M_jup = 9.54e-4
a_jup = 5.2  # AU
omega_jup = np.sqrt(G * M_sun / a_jup**3)

def jupiter_position(t):
    return np.array([
        a_jup * np.cos(omega_jup * t),
        a_jup * np.sin(omega_jup * t)
    ])

def acceleration(r, t):
    a = -G * M_sun * r / np.linalg.norm(r)**3
    rj = jupiter_position(t)
    a += -G * M_jup * (r - rj) / np.linalg.norm(r - rj)**3
    return a

def integrate(r0, v0, T, dt):
    r, v = r0.copy(), v0.copy()
    t = 0.0
    while t < T:
        k1v = acceleration(r, t)
        k1r = v

        k2v = acceleration(r + 0.5*dt*k1r, t + 0.5*dt)
        k2r = v + 0.5*dt*k1v

        k3v = acceleration(r + 0.5*dt*k2r, t + 0.5*dt)
        k3r = v + 0.5*dt*k2v

        k4v = acceleration(r + dt*k3r, t + dt)
        k4r = v + dt*k3v

        r += dt*(k1r + 2*k2r + 2*k3r + k4r)/6
        v += dt*(k1v + 2*k2v + 2*k3v + k4v)/6
        t += dt
    return r

def ftle(r0, v0, T, dt, eps=1e-6):
    r1 = integrate(r0, v0, T, dt)
    r2 = integrate(r0 + np.array([eps, 0]), v0, T, dt)
    d0 = eps
    dT = np.linalg.norm(r2 - r1)
    return (1/T) * np.log(dT / d0)

import matplotlib.pyplot as plt

N = 200
extent = 0.5  # AU around 10 AU
xs = np.linspace(10 - extent, 10 + extent, N)
ys = np.linspace(-extent, extent, N)

T = 200.0     # integration time
dt = 0.02

ftle_map = np.zeros((N, N))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        r0 = np.array([x, y])
        v0 = np.array([0.0, np.sqrt(G*M_sun/x)])  # circular guess
        ftle_map[j, i] = ftle(r0, v0, T, dt)
plt.imshow(
    ftle_map,
    extent=[xs[0], xs[-1], ys[0], ys[-1]],
    origin='lower',
    cmap='winter'
)
plt.colorbar(label='FTLE')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Lagrangian Stability Map near 10 AU')
plt.show()

