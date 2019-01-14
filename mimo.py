import numpy as np

def calc_bit(p, m):
    x_ = np.array(list(np.binary_repr(p, m)), dtype=float)
    x_ = x_ * 2. -1.
    return x_

def mimo(args):
    eps = 1e-12
    mode, num_symbol, m, n, sigma = args
    x = np.random.choice([1.+0j, -1.+0j], (m, num_symbol))
    h_re = np.random.normal(0, 1, (n,m))/np.sqrt(2)
    h_im = np.random.normal(0, 1, (n,m))/np.sqrt(2)*1j
    h_arr = h_re + h_im
    noi_re = np.random.normal(0, 1, (n, num_symbol))
    noi_im = np.random.normal(0, 1, (n, num_symbol))*1j
    noise = (noi_re + noi_im)*sigma
    rx = np.matmul(h_arr, x) + noise
    h_arr_h = h_arr.conj().T
    if mode == 'ZF':
        w = np.linalg.inv(h_arr_h @ h_arr + eps*np.eye(m)) @ h_arr_h
        est_x = np.matmul(w, rx)
    elif mode == 'MMSE':
        w = np.linalg.inv(h_arr @ h_arr_h + sigma*np.eye(n)) @ h_arr
        est_x = np.matmul(w.conj().T, rx)
    elif mode == 'MLD':
        num_p = np.arange(2**m)
        est_x = []
        for r in rx.T:
            cost_li = []
            for p in num_p:
                x_ = calc_bit(p, m)
                est_r  = h_arr @ x_
                cost_li.append(np.linalg.norm(r - est_r)**2)
            est_x_ = calc_bit(np.argmin(cost_li), m)
            est_x.append(est_x_)
        est_x = np.array(est_x).T
    err_rate = np.mean(1. * ((x > 0.) != (est_x > 0.)))
    return err_rate

def sim_loop(args, num_loop):
    err_arr = np.array([mimo(args) for _ in range(num_loop)])
    return np.mean(err_arr)

def calc_sigma(m, sn):
    snr = np.power(10., sn/10.)
    n0 = m/snr
    sigma = np.sqrt(n0/2.)
    return sigma

def main():
    num_loop = 1000
    num_symbol = 256
    m = 4
    n_li = [4, 8]
    sn_li = [0., 4., 40.]
    mode_li = ['ZF', 'MMSE', 'MLD']
    #mode_li = ['MLD']
    for mode in mode_li:
        for n in n_li:
            for sn in sn_li:
                sigma = calc_sigma(m, sn)
                args = (mode, num_symbol, m, n, sigma)
                error = sim_loop(args, num_loop)
                print(mode,n,sn,error)

if __name__ == '__main__':
    main()
