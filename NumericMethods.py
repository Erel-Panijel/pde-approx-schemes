import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def m_der(num, der):
    if der == '0':
        return np.roll(np.diag(np.ones(num)), -num) - np.roll(np.diag(np.ones(num)), num)
    elif der == '+-' or der == '-+':
        return np.diag(-2 * np.ones(num)) + np.roll(np.diag(np.ones(num)), -num) + np.roll(np.diag(np.ones(num)), num)
    elif der == '+':
        return np.diag(-np.ones(num)) + np.roll(np.diag(np.ones(num)), -num)
    elif der == '-':
        return np.diag(np.ones(num)) - np.roll(np.diag(np.ones(num)), num)
    else:
        raise ValueError('Please enter a valid operator')


def q_euler(num, k, h, der, theta=None):
    if der == 'fe':
        return np.identity(num) + 1/2 * k/h * m_der(num, '0')
    elif der == '+-' or der == '-+':
        return np.identity(num) + (k / (h ** 2)) * m_der(num, der)
    elif der == 'laxw':
        return np.identity(num) + 1/2 * k/h * m_der(num, '0') + 1/2 * (k/h) ** 2 * m_der(num, '+-')
    elif der == 'laxf':
        return np.identity(num) + 1/2 * k/h * m_der(num, '0') + 1/2 * m_der(num, '+-')
    elif der == 'dw':
        return np.identity(num) + k/h * m_der(num, '-')
    elif der == 'uw':
        return np.identity(num) + k/h * m_der(num, '+')
    elif der == 'be':
        return sp.linalg.circulant(sp.fft.irfft(1/sp.fft.rfft(q_euler(num, -k, h, 'fe')[:, 0])))
    elif der == 'lf':
        return np.block([[k/h * m_der(num, '0'), np.identity(num)], [np.identity(num), np.zeros((num, num))]])
    elif der == 'cn':
        return q_euler(num, k/2, h, 'be') @ q_euler(num, k/2, h, 'fe')
    elif der == 'theta':
        return q_euler(num, k * theta, h, 'be') @ q_euler(num, k * (1 - theta), h, 'fe')
    elif der == 'be2':
        return sp.linalg.circulant(sp.fft.irfft(1 / sp.fft.rfft(q_euler(num, -k, h, '+-')[:, 0])))


def euler_step(vec, q):
    return q @ vec


def lf_step(vec, prev, k, h, m):
    return prev + (k / h) * m @ vec, vec


def approx_scheme_advection(func, num, tf, k, scheme, loop=False, theta=None):
    n = int(tf / k)
    if scheme == 'lf':
        if loop:
            prev = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
            res = euler_step(prev, q_euler(num, k, 1 / num, 'fe'))
            m = m_der(num, '0')
            for i in range(n - 1):
                res, prev = lf_step(res, prev, k, 1 / num, m)
            return res
        else:
            res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
            res = np.hstack((euler_step(res, q_euler(num, k, 1 / num, 'fe')), res))
            q = q_euler(num, k, 1 / num, 'lf')
            res = euler_step(res, np.linalg.matrix_power(q, n))
            return res[:num]
    else:
        res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
        if theta:
            q = q_euler(num, k, 1 / num, 'theta', theta)
        elif scheme in ['fe', 'laxw', 'laxf', 'dw', 'uw', 'be', 'cn']:
            q = q_euler(num, k, 1 / num, scheme)
        else:
            raise ValueError('Invalid scheme. must be: fe, laxw, laxf, dw, uw, be, cn, theta or lf.')
        if loop:
            for i in range(n):
                res = euler_step(res, q)
        else:
            res = euler_step(res, np.linalg.matrix_power(q, n))
        return res


def fe_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'fe')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def laxw_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'laxw')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def laxf_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'laxf')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    # for i in range(n):
    #     res = euler_step(res, q)
    return res


def downwind_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'dw')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def upwind_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'uw')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def be_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'be')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def cn_method(func, num, tf, k):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'cn')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def theta_method(func, num, tf, k, theta):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'theta', theta)
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def lf_method(func, num, tf, k):
    n = int(tf / k) - 1
    prev = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    res = euler_step(prev, q_euler(num, k, 1 / num, '0'))
    m = m_der(num, '0')
    for i in range(n):
        res, prev = lf_step(res, prev, k, 1 / num, m)
    return res


def lf_method_v2(func, num, tf, k):
    n = int(tf / k) - 1
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    res = np.hstack((euler_step(res, q_euler(num, k, 1 / num, 'fe')), res))
    q = q_euler(num, k, 1 / num, 'lf')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res[:num]


def f_heat(vec):
    return 4 * (np.pi ** 2) * np.cos(2 * np.pi * vec) - 2 * np.pi * np.sin(2 * np.pi * vec)


def fe_method_2nd(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    q = q_euler(num, k, 1 / num, '+-')
    for i in range(n):
        res = euler_step(res, q) + k * f_heat((i * k) + xvals)
    return res


def fe_method_2nd_hom(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    q = q_euler(num, k, 1 / num, '+-')
    res = np.linalg.matrix_power(q, n) @ res
    return res


def be_method_2nd(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in np.linspace(0, 1, num, endpoint=False)])
    q = q_euler(num, k, 1 / num, 'be2')
    for i in range(n):
        res = euler_step(res + k * f_heat((i * k) + xvals), q)
    return res


def be_method_2nd_hom(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    q = q_euler(num, k, 1 / num, 'be2')
    res = euler_step(res, np.linalg.matrix_power(q, n))
    return res


def cn_method_2nd(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    qfe = q_euler(num, k/2, 1/num, '+-')
    qbe = q_euler(num, k/2, 1/num, 'be2')
    for i in range(n):
        res = euler_step(euler_step(res, qfe) + k * (f_heat(k * i + xvals) + f_heat(k * (i + 1) + xvals))/2, qbe)
    return res


def cn_method_2nd_hom(func, num, tf, k, xvals):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    qfe = q_euler(num, k/2, 1/num, '+-')
    qbe = q_euler(num, k/2, 1/num, 'be2')
    res = euler_step(res, np.linalg.matrix_power(qbe @ qfe, n))
    # for i in range(n):
    #     res = euler_step(euler_step(res, qfe), qbe)
    return res


def theta_method_2nd(func, num, tf, k, xvals, theta):
    n = int(tf / k)
    res = np.array([func(i) for i in xvals])
    qfe = q_euler(num, k * (1 - theta), 1/num, '+-')
    qbe = q_euler(num, k * theta, 1/num, 'be2')
    for i in range(n):
        res = euler_step(euler_step(res, qfe) + k * f_heat((i * k) + xvals), qbe)
    return res


def lf_method_2nd(func, num, tf, k):
    n = int(tf / k) - 1
    xval = np.linspace(0, 1, num, endpoint=False)
    prev = np.array([func(i) for i in xval])
    res = euler_step(prev, q_euler(num, k, 1 / num, '+-')) + k * f_heat(xval)
    m = m_der(num, '+-')
    for i in range(n):
        res, prev = lf_step(res, prev, 2 * k, (1 / num) ** 2, m)
        res += 2 * k * f_heat((i + 1) * k + xval)
    return res


def errorh(vec1, vec2, h):
    return np.sqrt(h) * (np.linalg.norm(vec1 - vec2))


# # advection equation euler method
# errs = []
# pts = 8
# for n in range(7):
#     pts = 2 * pts
#     dx = 1 / pts
#     xval = np.linspace(0, 1, pts, endpoint=False)
#     yval = np.array([np.cos(2 * np.pi * i) for i in xval])
#     ycal = fe_method(lambda x: np.cos(2 * np.pi * x), pts, 5, 1/2 * dx**2)
#     # print(pts)
#     # plt.figure(n)
#     # plt.plot(xval, yval)
#     # plt.plot(xval, ycal)
#     errs.append([dx, errorh(yval, ycal, dx)])
# errs = np.transpose(np.array(errs))
# plt.plot(errs[0], errs[1], 'o-')
# plt.xscale('log')
# plt.xlabel(r'$\Delta x$')
# plt.yscale('log')
# plt.ylabel(r'$|| E_h ||$')
# plt.legend([rf'$k = \lambda h$, slope: {np.polyfit(-np.log(errs[0]), -np.log(errs[1]), 1)[0]:.5f}'])
# plt.title('Advection equation using the FE method')
# plt.show()


# advection equation leap frog method
errs = []
pts = 8
for a in range(5):
    pts = 2 * pts
    dx = 1 / pts
    # dt_lin = 1/6 * dx
    dt_sqr = 1/2 * dx ** 2
    # tf_lin = dt_lin * int(5/dt_lin)
    tf_sqr = dt_sqr * int(5/dt_sqr)
    xval = np.linspace(0, 1, pts, endpoint=False)
    # yval1 = np.array([np.cos(2 * np.pi * i) for i in xval]) * np.exp(-4 * np.pi**2 * tf_lin)
    yval1 = np.array([np.cos(2 * np.pi * (i + tf_sqr)) for i in xval])
    ycal1 = fe_method_2nd(lambda x: np.cos(2 * np.pi * x), pts, 5, dt_sqr, xval)
    print(pts, 1)
    # ycal2 = theta_method_2nd(lambda x: np.cos(2 * np.pi * x), pts, 5, dt_sqr, xval, 0.5)
    # print(pts, 2)
    errs.append([dx, errorh(yval1, ycal1, dx)])
errs = np.transpose(np.array(errs))
coeffs1 = np.polyfit(-np.log(errs[0]), -np.log(errs[1]), 1)
# coeffs2 = np.polyfit(-np.log(errs[0]), -np.log(errs[2]), 1)
p1 = np.poly1d(coeffs1)
# p2 = np.poly1d(coeffs2)
plt.scatter(errs[0], errs[1], marker='o')
# plt.scatter(errs[0], errs[2], marker='*')
plt.plot(errs[0], np.exp(-p1(-np.log(errs[0]))), '-c')
# plt.plot(errs[0], np.exp(-p2(-np.log(errs[0]))), '-y')
plt.xscale('log')
plt.xlabel(r'$\Delta x$')
plt.yscale('log')
plt.ylabel(r'$|| E_h ||$')
plt.legend([rf'$k = \lambda h$, slope: {coeffs1[0]:.5f}'])
plt.title('Heat equation using the Forward-Euler method')
plt.show()


# # heat equation euler method
# errs = []
# pts = 8
# for a in range(5):
#     pts = 2 * pts
#     dx = 1 / pts
#     dt_sqr = 1/6 * dx**2
#     tf_sqr = dt_sqr * int(5/dt_sqr)
#     xval = np.linspace(0, 1, pts, endpoint=False)
#     yval = np.array([np.cos(2 * np.pi * (i + tf_sqr)) for i in xval])
#     ycal = cn_method_2nd(lambda x: np.cos(2 * np.pi * x), pts, 5, dt_sqr, xval)
#     print(a)
#     errs.append([dx, errorh(yval, ycal, dx)])
# errs = np.transpose(np.array(errs))
# coeffs = np.polyfit(-np.log(errs[0]), -np.log(errs[1]), 1)
# p1 = np.poly1d(coeffs)
# plt.scatter(errs[0], errs[1], marker='o')
# plt.plot(errs[0], np.exp(-p1(-np.log(errs[0]))), '-c')
# plt.xscale('log')
# plt.xlabel(r'$\Delta x$')
# plt.yscale('log')
# plt.ylabel(r'$|| E_h ||$')
# plt.legend([rf'$k = \lambda h$, slope: {coeffs[0]:.5f}'])
# plt.title('Heat equation using the FE method')
# plt.show()


# # heat equation leap frog method
# errs = []
# pts = 8
# for a in range(4):
#     pts = 2 * pts
#     dx = 1 / pts
#     xval = np.linspace(0, 1, pts, endpoint=False)
#     yval = np.array([np.cos(2 * np.pi * i) for i in xval])
#     ycal = lf_method_2nd(lambda x: np.cos(2 * np.pi * x), pts, 5, 1/4 * dx**2)
#     errs.append([dx, errorh(yval, ycal, dx)])
# errs = np.transpose(np.array(errs))
# plt.plot(errs[0], errs[1], 'o-')
# plt.xscale('log')
# plt.xlabel(r'$\Delta x$')
# plt.yscale('log')
# plt.ylabel(r'$|| E_h ||$')
# plt.legend([rf'$k = \lambda h^2$, slope: {np.polyfit(-np.log(errs[0]), -np.log(errs[1]), 1)[0]:.5f}'])
# plt.title('Heat equation using the LF method')
# plt.show()
