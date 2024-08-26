import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

def initialize_temperature(nx, dx, L):
    """初期温度分布を生成する関数（方形波）"""
    temp = np.zeros(nx)
    for i in range(nx):
        x = dx * i
        temp[i] = 5 * (x / (L / 2)) if i < nx // 2 else 5 * (L - x) / (L / 2)
    return temp

def update_temperature(temp, nx, alpha, dt, dx):
    """温度分布を更新する関数"""
    new_temp = temp.copy()
    for i in range(1, nx - 1):
        new_temp[i] = temp[i] + alpha * dt / dx**2 * (temp[i+1] - 2 * temp[i] + temp[i-1])
    return new_temp

def analytical_solution(X, t, L, alpha, n_terms=1000):
    """解析的解を計算する関数"""
    solution = np.zeros_like(X)
    for n in range(1, n_terms + 1):
        bn = 20 / (n * np.pi)**2 * (1 - (-1)**n)
        solution += bn * np.sin(n * np.pi * X / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)
    return solution

def plot_and_save_fig(X, nt, dt, ana_sol, num_sol, file_name):
    """2つ解のヒートマップをプロットする関数"""
    T = np.linspace(0, nt*dt, nt)

    plt.figure(figsize=(18, 18))

    # 解析解
    plt.subplot(311)
    plt.contourf(T, X, ana_sol.T, 20, cmap='hot')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Analytical Solution')

    #  数値解
    plt.subplot(312)
    plt.contourf(T, X, num_sol.T, 20, cmap='hot')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Numerical Solution')

    # 全体の最大値を取得
    max_val = max(np.max(ana_sol), np.max(num_sol))
    # 特定の時点での比較プロット
    for i, t in enumerate([0.0,0.25,0.5,0.75]):
        plt.subplot(3, 4, 9+i)
        plt.plot(X, ana_sol[int(nt*t), :], 'b--', label='Analytical')
        plt.plot(X, num_sol[int(nt*t), :], 'r', label='Numerical')
        plt.title(f't = {nt*dt*t}')
        plt.ylim(0, max_val)
        plt.xlabel('x')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig(file_name)


def save_dataset(X, nt, dt, analytical_sol, file_name):
    T = np.linspace(0, nt*dt, nt)
    data = {'x':X, 't':T, 'u':analytical_sol.T}
    scipy.io.savemat(file_name, data)


if __name__ == '__main__':
    # ディレクトリの作成
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # パラメータ設定
    nx = 100
    nt = 2500
    dt = 0.004
    alpha = 1
    L = 10
    dx = L / nx

    # 初期温度分布のプロット
    X = np.linspace(0, L, nx)
    temp = initialize_temperature(nx, dx, L)

    # 解析解
    analytical_sol = np.zeros((nt, nx))
    analytical_sol[0, :] = temp
    for ti in range(1, nt):
        analytical_sol[ti, :] = analytical_solution(X, ti*dt, L, alpha)
    
    # 数値解析
    numerical_sol = np.zeros((nt, nx))
    numerical_sol[0, :] = temp
    for ti in range(1, nt):
        temp = update_temperature(temp, nx, alpha, dt, dx)
        numerical_sol[ti, :] = temp

    print(f'{analytical_sol.shape=}')
    print(f'{numerical_sol.shape=}')
    T = np.linspace(0, nt*dt, nt)
    print(f'{X.shape=}')
    print(f'{X.max()=}')
    print(f'{X.min()=}')
    print(f'{T.shape=}')
    print(f'{T.max()=}')
    print(f'{T.min()=}')

    plot_and_save_fig(X, nt, dt, analytical_sol, numerical_sol,
                      os.path.join(output_dir, 'fig.png'))
    
    # dataset用
    L_d = nt*dt
    nt_d = 200
    dt_d = L / nt_d
    T_dataset = np.linspace(0, L, nt_d)
    ana_sol_dataset = np.zeros((nt_d, nx))
    ana_sol_dataset[0, :] = temp
    for ti in range(1, nt_d):
        ana_sol_dataset[ti, :] = analytical_solution(X, ti*dt_d, L_d, alpha)
  
    plt.figure(figsize=(8, 6))
    plt.contourf(X, T_dataset, ana_sol_dataset, 20, cmap='hot')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('heat_diffusion')
    plt.savefig(os.path.join(output_dir, 'dataset.png'))
    
    save_dataset(X, nt, dt, ana_sol_dataset,
                 os.path.join(output_dir, '../../../data/heat_diffusion.mat'))