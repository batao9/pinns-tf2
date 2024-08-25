import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

def analysis_solution(X_arr, T_arr, mu, sigma, alpha=0.01):
    """
    熱拡散方程式の解析解を計算する関数。

    :param x: 空間の配列
    :param t: 時間の配列
    :param alpha: 熱拡散率 (デフォルトは0.01)
    :return: 解析解の配列
    """

    u = np.zeros((len(T_arr), len(X_arr)))

    for t in range(len(T_arr)):
        u[t, :] = np.exp(-0.5 * ((X_arr - mu) / (sigma * np.sqrt(1 + 2 * alpha * T_arr[t]))) ** 2) \
                  / np.sqrt(1 + 2 * alpha * T_arr[t])

    return u

def save_heat_map(X, T, u, file_name):
    """
    熱マップを保存する関数。

    :param X: 空間の配列
    :param T: 時間の配列
    :param u: 解析解の配列
    :param file_name: 保存するファイルの名前
    """
    plt.figure(figsize=(8, 6))
    plt.contourf(X, T, u, 20, cmap='hot')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('heat_diffusion')
    plt.savefig(file_name)
    plt.show()
    return 


def save_to_matfile(x, t, u, file_name):
    """
    x, t, uをmatファイルとして保存する関数。

    :param x: 空間の配列
    :param t: 時間の配列
    :param u: 解析解の配列
    :param file_name: 保存するファイルの名前
    """
    data = {'x': x, 't': t, 'u': u}
    scipy.io.savemat(file_name, data)


def heat_diffusion_dataset():
    """
    熱拡散方程式に基づくデータセットを作成する関数。

    :return: 作成されたデータセット
    """
    current_dir = os.path.dirname(__file__)

    # params
    alpha = 0.01  # 熱拡散率
    L = 1.0  # 空間の長さ
    T = 1.0  # 時間の長さ
    Nx = 200  # 空間の分割数
    Nt = 1000  # 時間の分割数
    dx = L / (Nx - 1)  # 空間ステップ
    dt = T / (Nt - 1)  # 時間ステップ
    mu = L / 2.0
    sigma = 1

    # 空間と時間の範囲を定義
    X_arr = np.linspace(0, L, Nx)
    T_arr = np.linspace(0, T, Nt)

    # 初期条件を設定
    u0 = np.exp(-0.5 * ((X_arr - mu) / sigma) ** 2)

    u = analysis_solution(X_arr, T_arr, mu, sigma, alpha)

    print(f'{X_arr.shape=}')
    print(f'{T_arr.shape=}')
    print(f'{u.shape=}')
    
    # heat map保存
    output_dir = os.path.join(current_dir, 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_heat_map(X_arr, T_arr, u, 
                  os.path.join(output_dir, 'heat_diffusion_heatmap.png'))

    
    # mat file 保存
    save_to_matfile(X_arr, T_arr, u,
                    os.path.join(current_dir, '../../data/heat_duffusion.mat'))



if __name__ == '__main__':
    heat_diffusion_dataset()