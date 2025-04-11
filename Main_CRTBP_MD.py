import numpy as np
import random
import math
import warnings
import time
import scipy
import scipy.io as scio
from scipy.stats import norm
from scipy.integrate import solve_ivp
from module_crtbp import CRTBP_dynamics, CRTBP, CRTBP_Acc_dynamics, CRTBP_STM_Jacobi, CRTBP_STT_Jacobi
from module_maneuver_detection import generate_scenario_single_angle, generate_maneuver_detection_confidences_adaptive, pad_and_stack_arrays
from module_reachable_set import measurement_model

warnings.filterwarnings("ignore")
RelTol = 10 ** -12
AbsTol = 10 ** -12
miuE = 398600.435436096
Re = 6378.1366
mu = 0.0121505839705277
unitL = 384400
unitV = 1.02454629434750
unitT = 375190.464423878


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_initial_states():
    """Generate the initial states for both target and observer"""
    period = np.zeros(2)
    """Target"""
    data = scipy.io.loadmat("./data/Scenario/NRHO_Stable_Data.mat")
    period[0] = data["period"][0, 0]
    xT0 = data["x0"].T[0]
    """Observer"""
    data = scipy.io.loadmat("./data/Scenario/NRHO_9_2_Data.mat")
    xO0 = data["x0"].T[0]
    period[1] = data["period"][0, 0]
    """Return results"""
    return xT0, xO0, period

def get_scenario_parameters(
        param: dict,
):
    """Generate parameters for simulation"""
    """Initial states"""
    xT0, xO0, period = get_initial_states()
    t0 = 0.0
    tf = 3.0 * period[0]  # 1 revolution
    """Dynamics setting"""
    f = lambda t, x: CRTBP_dynamics(t, x, mu)
    ft = lambda x, tau: CRTBP(x, tau, mu)
    fu = lambda t, x, u: CRTBP_Acc_dynamics(t, x, u, mu)
    Jacobi = lambda x: CRTBP_STM_Jacobi(x, mu)
    Jacobi2 = lambda x: CRTBP_STT_Jacobi(x, mu)
    errR = 1e0 / unitL * param["P0"]
    errV = 1e-4 / unitV * param["P0"]
    P0 = np.diag(np.array([
        errR ** 2, errR ** 2, errR ** 2,
        errV ** 2, errV ** 2, errV ** 2,
    ]))
    std_angle = 5 / 3600 * math.pi / 180 * param["R"]
    R = np.diag(
        np.ones([2]) * (std_angle ** 2),
    )
    DIMx = 6
    DIMz = 2  # dimension of the measurement
    DIMxz = DIMx + DIMz
    Pw = np.zeros([DIMxz, DIMxz])
    Pw[:DIMx, :DIMx] = P0
    Pw[DIMx:, DIMx:] = R
    """Propagate the nominal orbit"""
    t_eval = [t0, tf]
    sol = solve_ivp(f, [t0, tf], xT0, args=(), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xTf = sol.y.T[-1]
    t_eval = [t0, 0.85 * period[1]]
    sol = solve_ivp(f, [t0, tf], xO0, args=(), method='RK45',
                    t_eval=t_eval, max_step=np.inf, rtol=RelTol, atol=AbsTol)
    xOf = sol.y.T[-1]
    """Return results"""
    return xT0, xTf, t0, tf, xO0, xOf, f, ft, fu, Jacobi, Jacobi2, P0, R, Pw, DIMx, DIMz

def single_angle_case_one_run(
        ifManeuver: bool,
        dvmax: float,
        param: dict,
        ifSave: bool,
) -> tuple[np.array, np.array, float, float, np.array, np.array]:
    """
    Maneuver detection using single angle measurement (one-run simulation)
    """
    """Parameters setting"""
    setup_seed(42)
    xT0, xTf, t0, tf, xO0, xOf, f, ft, fu, Jacobi, Jacobi2, P0, R, Pw, DIMx, DIMz = get_scenario_parameters(param=param)
    """Define maximal maneuver"""
    if ifManeuver:
        dv = dvmax
    else:
        dv = 0.0
    """Begin one-run simulation"""
    Nc, Ns, Na = 100, 11, 5
    time_costs = np.zeros([7])

    """Randomly generate the simulation scenario"""
    """Add maneuver"""
    alpha = np.random.uniform(
        low=-math.pi / 2, high=math.pi / 2, size=None
    )
    beta = np.random.uniform(
        low=-math.pi, high=math.pi, size=None
    )
    dvx = dv * math.cos(alpha) * math.cos(beta)
    dvy = dv * math.cos(alpha) * math.sin(beta)
    dvz = dv * math.sin(alpha)
    xT0r, xT0e, xTfe, dxT0, Measurement, angle_errors = generate_scenario_single_angle(
        xT0=xT0, xOf=xOf, t0=t0, tf=tf, f=f, P0=P0, R=R, dvx=dvx, dvy=dvy, dvz=dvz,
    )
    nominal_angle = measurement_model(xTfe, xOf)

    """Reachable set-based maneuver detection"""
    print("======== Begin reachable set-based maneuver detection ========")
    """Maneuver detection via determining RS"""
    confidences_list = list()

    """Adaptive method with minimization strategy (no initial guess)"""
    print("======== Adaptive method with minimization strategy (no initial guess) ========")
    confidences1, probability1, _, time_cost1 = generate_maneuver_detection_confidences_adaptive(
        xT0=xT0e, xTf=xTfe, xOf=xOf, t0=t0, tf=tf, measurement=Measurement, Nc=Nc, Ns=Ns, Na=Na,
        P0=P0, R=R, f=ft, param=param, strategy=3, if_use_guess=False,
    )
    confidences_list.append(confidences1)

    """Adaptive method with recursive strategy"""
    print("======== Adaptive method with recursive strategy ========")
    confidences2, probability2, _, time_cost2 = generate_maneuver_detection_confidences_adaptive(
        xT0=xT0e, xTf=xTfe, xOf=xOf, t0=t0, tf=tf, measurement=Measurement, Nc=Nc, Ns=Ns, Na=Na,
        P0=P0, R=R, f=ft, param=param, strategy=2,
    )
    confidences_list.append(confidences2)

    """RS-all"""
    time_costs[0] = float(time_cost1["build_map"])
    time_costs[1] = float(time_cost2["build_map"])
    time_costs[2] = float(np.sum(time_cost1["RS"]))
    time_costs[3] = float(np.sum(time_cost2["RS"]))

    confidences_array, confidences_length = pad_and_stack_arrays(array_list=confidences_list)
    probability = np.array([probability1, probability2])

    """Save data"""
    if ifSave:
        """.mat file"""
        if ifManeuver:
            file_name = "./data/Single/single_angle_M_one_run.mat"
        else:
            file_name = "./data/Single/single_angle_NM_one_run.mat"
        scio.savemat(
            file_name, {
                "xT0r": xT0r,
                "xT0e": xT0e,
                "xO0": xO0,
                "xOf": xOf,
                "t0": t0,
                "tf": tf,
                "dvmax": dvmax,
                "dv": dv,
                "Measurement": Measurement,
                "confidences_array": confidences_array,
                "confidences_length": confidences_length,
                "probability": probability,
                "time_costs": time_costs,
                "nominal_angle": nominal_angle,
            },
        )
    """Return data"""
    return confidences_array, probability, p1, dz, time_costs, dv


if __name__ == "__main__":
    """One run simulation for the single-angle case"""
    param = {
        "P0": 1,
        "R": 1,
        "order": 5,
    }
    dvmax = 5.0e-4 / unitV
    ifManeuver = True
    confidences, probability, p1, dz, time_costs, dv = single_angle_case_one_run(
        ifManeuver=ifManeuver,
        dvmax=dvmax,
        param=param,
        ifSave=True,
    )
