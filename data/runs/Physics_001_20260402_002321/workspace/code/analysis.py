"""
MATBG Superfluid Stiffness Analysis
====================================
Analyzes the superfluid stiffness of Magic-Angle Twisted Bilayer Graphene (MATBG)
across three experimental scenarios:
  1. Carrier density dependence (quantum geometry enhancement)
  2. Temperature dependence (power-law behavior / gap symmetry)
  3. Current dependence (Ginzburg-Landau vs Meissner responses)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import os

# ── Output paths ──────────────────────────────────────────────────────────────
WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_001_20260402_002321"
IMG_DIR   = os.path.join(WORKSPACE, "report", "images")
OUT_DIR   = os.path.join(WORKSPACE, "outputs")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 2,
})

# =============================================================================
# DATASET
# =============================================================================

# --- File 1: Carrier-density dependence -------------------------------------
n_eff = np.array([5.00000000e+14, 5.91836735e+14, 6.83673469e+14, 7.75510204e+14,
 8.67346939e+14, 9.59183673e+14, 1.05102041e+15, 1.14285714e+15,
 1.23469388e+15, 1.32653061e+15, 1.41836735e+15, 1.51020408e+15,
 1.60204082e+15, 1.69387755e+15, 1.78571429e+15, 1.87755102e+15,
 1.96938776e+15, 2.06122449e+15, 2.15306122e+15, 2.24489796e+15,
 2.33673469e+15, 2.42857143e+15, 2.52040816e+15, 2.61224490e+15,
 2.70408163e+15, 2.79591837e+15, 2.88775510e+15, 2.97959184e+15,
 3.07142857e+15, 3.16326531e+15, 3.25510204e+15, 3.34693878e+15,
 3.43877551e+15, 3.53061224e+15, 3.62244898e+15, 3.71428571e+15,
 3.80612245e+15, 3.89795918e+15, 3.98979592e+15, 4.08163265e+15,
 4.17346939e+15, 4.26530612e+15, 4.35714286e+15, 4.44897959e+15,
 4.54081633e+15, 4.63265306e+15, 4.72448980e+15, 4.81632653e+15,
 4.90816327e+15, 5.00000000e+15])

D_s_conv = np.array([1.14642368e+09, 1.24696564e+09, 1.34039778e+09, 1.42782172e+09,
 1.51002634e+09, 1.58760949e+09, 1.66103385e+09, 1.73066529e+09,
 1.79679961e+09, 1.85967941e+09, 1.91950426e+09, 1.97643991e+09,
 2.03062413e+09, 2.08217148e+09, 2.13117727e+09, 2.17772057e+09,
 2.22186633e+09, 2.26366730e+09, 2.30316590e+09, 2.34039589e+09,
 2.37538376e+09, 2.40815000e+09, 2.43871016e+09, 2.46707582e+09,
 2.49325536e+09, 2.51725466e+09, 2.53907765e+09, 2.55872678e+09,
 2.57620348e+09, 2.59250847e+09, 2.60764211e+09, 2.62160464e+09,
 2.63439640e+09, 2.64601800e+09, 2.65647044e+09, 2.66575522e+09,
 2.67387435e+09, 2.68083038e+09, 2.68662640e+09, 2.69126601e+09,
 2.69475331e+09, 2.69709289e+09, 2.69828981e+09, 2.69834957e+09,
 2.69727810e+09, 2.69508174e+09, 2.69176722e+09, 2.68734165e+09,
 2.68181248e+09, 2.67518747e+09])

D_s_geom = np.array([4.91324433e+09, 5.34628159e+09, 5.74627790e+09, 6.11923597e+09,
 6.46968559e+09, 6.80104047e+09, 7.11599637e+09, 7.41669926e+09,
 7.70485534e+09, 7.98182318e+09, 8.24866141e+09, 8.50618546e+09,
 8.75503198e+09, 8.99569258e+09, 9.22854793e+09, 9.45388790e+09,
 9.67192698e+09, 9.88281486e+09, 1.00866482e+10, 1.02831122e+10,
 1.04723646e+10, 1.06546435e+10, 1.08301850e+10, 1.09992225e+10,
 1.11619857e+10, 1.13187004e+10, 1.14695875e+10, 1.16148630e+10,
 1.17547375e+10, 1.18894159e+10, 1.20190975e+10, 1.21439763e+10,
 1.22642406e+10, 1.23800735e+10, 1.24916525e+10, 1.25991504e+10,
 1.27027346e+10, 1.28025679e+10, 1.28988080e+10, 1.29916079e+10,
 1.30811160e+10, 1.31674761e+10, 1.32508276e+10, 1.33313054e+10,
 1.34090404e+10, 1.34841593e+10, 1.35567847e+10, 1.36270356e+10,
 1.36950268e+10, 1.37608697e+10])

D_s_exp_hole = np.array([3.85604343e+10, 4.24265821e+10, 4.52423238e+10, 4.93808532e+10,
 5.19704020e+10, 5.57956448e+10, 5.90341377e+10, 6.32891365e+10,
 6.66534759e+10, 7.05280091e+10, 7.38961211e+10, 7.86183663e+10,
 8.14731800e+10, 8.59709197e+10, 8.94166552e+10, 9.42492950e+10,
 9.76904386e+10, 1.02049782e+11, 1.06107094e+11, 1.10086450e+11,
 1.14387728e+11, 1.18191820e+11, 1.22351351e+11, 1.26525799e+11,
 1.30167348e+11, 1.34355941e+11, 1.37852551e+11, 1.42465133e+11,
 1.45881941e+11, 1.50437589e+11, 1.54495487e+11, 1.58821246e+11,
 1.63023454e+11, 1.67471364e+11, 1.71755524e+11, 1.75901845e+11,
 1.80239352e+11, 1.84604118e+11, 1.88613997e+11, 1.92878937e+11,
 1.96996621e+11, 2.01188574e+11, 2.05285732e+11, 2.09374636e+11,
 2.13632989e+11, 2.17696229e+11, 2.21767737e+11, 2.25876875e+11,
 2.29888452e+11, 2.33911617e+11])

D_s_exp_electron = np.array([3.66324126e+10, 4.03052529e+10, 4.29802076e+10, 4.69118005e+10,
 4.93718819e+10, 5.30058625e+10, 5.60824308e+10, 6.01246797e+10,
 6.33208021e+10, 6.70016086e+10, 7.02013150e+10, 7.46874480e+10,
 7.73995210e+10, 8.16723737e+10, 8.49458224e+10, 8.95368203e+10,
 9.28059167e+10, 9.69472943e+10, 1.00801739e+11, 1.04582127e+11,
 1.08668342e+11, 1.12282229e+11, 1.16233784e+11, 1.20199509e+11,
 1.23658981e+11, 1.27638144e+11, 1.30959923e+11, 1.35341876e+11,
 1.38587844e+11, 1.42915709e+11, 1.46750713e+11, 1.50880184e+11,
 1.54872282e+11, 1.59097796e+11, 1.63167748e+11, 1.67106753e+11,
 1.71247384e+11, 1.75373912e+11, 1.79183298e+11, 1.83234991e+11,
 1.87146790e+11, 1.91129146e+11, 1.95021445e+11, 1.98905904e+11,
 2.02951340e+11, 2.06811417e+11, 2.10679350e+11, 2.14583031e+11,
 2.18394030e+11, 2.22216036e+11])

# --- File 2: Temperature dependence -----------------------------------------
T = np.array([0.00000000e+00, 1.21212121e-02, 2.42424242e-02, 3.63636364e-02,
 4.84848485e-02, 6.06060606e-02, 7.27272727e-02, 8.48484848e-02,
 9.69696970e-02, 1.09090909e-01, 1.21212121e-01, 1.33333333e-01,
 1.45454545e-01, 1.57575758e-01, 1.69696970e-01, 1.81818182e-01,
 1.93939394e-01, 2.06060606e-01, 2.18181818e-01, 2.30303030e-01,
 2.42424242e-01, 2.54545455e-01, 2.66666667e-01, 2.78787879e-01,
 2.90909091e-01, 3.03030303e-01, 3.15151515e-01, 3.27272727e-01,
 3.39393939e-01, 3.51515152e-01, 3.63636364e-01, 3.75757576e-01,
 3.87878788e-01, 4.00000000e-01, 4.12121212e-01, 4.24242424e-01,
 4.36363636e-01, 4.48484848e-01, 4.60606061e-01, 4.72727273e-01,
 4.84848485e-01, 4.96969697e-01, 5.09090909e-01, 5.21212121e-01,
 5.33333333e-01, 5.45454545e-01, 5.57575758e-01, 5.69696970e-01,
 5.81818182e-01, 5.93939394e-01, 6.06060606e-01, 6.18181818e-01,
 6.30303030e-01, 6.42424242e-01, 6.54545455e-01, 6.66666667e-01,
 6.78787879e-01, 6.90909091e-01, 7.03030303e-01, 7.15151515e-01,
 7.27272727e-01, 7.39393939e-01, 7.51515152e-01, 7.63636364e-01,
 7.75757576e-01, 7.87878788e-01, 8.00000000e-01, 8.12121212e-01,
 8.24242424e-01, 8.36363636e-01, 8.48484848e-01, 8.60606061e-01,
 8.72727273e-01, 8.84848485e-01, 8.96969697e-01, 9.09090909e-01,
 9.21212121e-01, 9.33333333e-01, 9.45454545e-01, 9.57575758e-01,
 9.69696970e-01, 9.81818182e-01, 9.93939394e-01, 1.00606061e+00,
 1.01818182e+00, 1.03030303e+00, 1.04242424e+00, 1.05454545e+00,
 1.06666667e+00, 1.07878788e+00, 1.09090909e+00, 1.10303030e+00,
 1.11515152e+00, 1.12727273e+00, 1.13939394e+00, 1.15151515e+00,
 1.16363636e+00, 1.17575758e+00, 1.18787879e+00, 1.20000000e+00])
T_c = 1.0
D_s0 = 100.0

D_s_bcs = np.array([100., 99.98530718, 99.94122991, 99.8677697, 99.76492951,
 99.63271377, 99.47112834, 99.28018052, 99.059879, 98.81023395,
 98.5312569, 98.22296079, 97.88535997, 97.51847013, 97.12230838,
 96.69689318, 96.24224441, 95.75838332, 95.24533251, 94.70311596,
 94.131759, 93.53128831, 92.90173192, 92.24311919, 91.55548082,
 90.83884886, 90.09325669, 89.31873903, 88.51533194, 87.68307281,
 86.82200038, 85.93215469, 85.01357713, 84.06631038, 83.09039843,
 82.08588656, 81.05282136, 79.99125074, 78.90122389, 77.78279132,
 76.63600481, 75.46091744, 74.25758358, 73.02605886, 71.7664002,
 70.47866578, 69.16291506, 67.81920877, 66.44760892, 65.04817877,
 63.62098284, 62.16608686, 60.68355783, 59.17346395, 57.63587468,
 56.0708607, 54.47849394, 52.85884756, 51.21199597, 49.53801481,
 47.83698097, 46.10897259, 44.35406905, 42.57235096, 40.76390016,
 38.92879972, 37.06713395, 35.17898837, 33.26444974, 31.32360604,
 29.35654647, 27.36336145, 25.34414259, 23.29898276, 21.22797602,
 19.13121766, 17.00880419, 14.86083331, 12.68740394, 10.48861619,
 8.26457134, 6.01537186, 3.74112139, 1.44192472, 0.,
 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0.])

D_s_nodal = np.array([100., 98.78787879, 97.57575758, 96.36363636, 95.15151515,
 93.93939394, 92.72727273, 91.51515152, 90.3030303, 89.09090909,
 87.87878788, 86.66666667, 85.45454545, 84.24242424, 83.03030303,
 81.81818182, 80.60606061, 79.39393939, 78.18181818, 76.96969697,
 75.75757576, 74.54545455, 73.33333333, 72.12121212, 70.90909091,
 69.6969697, 68.48484848, 67.27272727, 66.06060606, 64.84848485,
 63.63636364, 62.42424242, 61.21212121, 60., 58.78787879,
 57.57575758, 56.36363636, 55.15151515, 53.93939394, 52.72727273,
 51.51515152, 50.3030303, 49.09090909, 47.87878788, 46.66666667,
 45.45454545, 44.24242424, 43.03030303, 41.81818182, 40.60606061,
 39.39393939, 38.18181818, 36.96969697, 35.75757576, 34.54545455,
 33.33333333, 32.12121212, 30.90909091, 29.6969697, 28.48484848,
 27.27272727, 26.06060606, 24.84848485, 23.63636364, 22.42424242,
 21.21212121, 20., 18.78787879, 17.57575758, 16.36363636,
 15.15151515, 13.93939394, 12.72727273, 11.51515152, 10.3030303,
 9.09090909, 7.87878788, 6.66666667, 5.45454545, 4.24242424,
 3.03030303, 1.81818182, 0.60606061, 0., 0.,
 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0.])

# Note: the BCS and nodal arrays have 95 entries, power_n3 has 90 entries.
# T has 100 entries. We build aligned versions by trimming T to match each array.
def T_for(arr):
    """Return the temperature sub-array matching the length of arr."""
    return T[:len(arr)]

D_s_power_n2 = D_s_bcs.copy()   # same array as BCS in this dataset

D_s_power_n2_5 = np.array([100., 99.98283379, 99.93134106, 99.84554288, 99.72546524,
 99.57113906, 99.38259922, 99.15988456, 98.90303787, 98.61210592,
 98.28713946, 97.92819321, 97.53532588, 97.10860017, 96.64808279,
 96.15384445, 95.62595988, 95.06450786, 94.4695712, 93.84123676,
 93.17959549, 92.48474239, 91.75677653, 90.99580108, 90.20192332,
 89.37525462, 88.51591047, 87.62401048, 86.69967838, 85.74304204,
 84.75423346, 83.73338878, 82.68064831, 81.59615654, 80.48006212,
 79.3325179, 78.1536809, 76.94371238, 75.70277777, 74.43104673,
 73.1286931, 71.79589495, 70.43283456, 69.03969841, 67.61667723,
 66.16396599, 64.68176391, 63.17027448, 61.62970545, 60.06026888,
 58.46218114, 56.83566295, 55.18093932, 53.49823966, 51.78779772,
 50.04985167, 48.28464407, 46.4924219, 44.67343658, 42.82794402,
 40.95620461, 39.05848326, 37.13504941, 35.18617703, 33.21214469,
 31.21323551, 29.18973722, 27.14194219, 25.07014741, 22.97465456,
 20.85576998, 18.71380478, 16.54907481, 14.36190073, 12.15260806,
 9.9215272, 7.66899347, 5.39534715, 3.10093354, 0.78610296,
 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0.])

D_s_power_n3 = np.array([100., 99.98036041, 99.92144337, 99.823263, 99.68583738,
 99.50918857, 99.2933426, 98.93832946, 98.54418213, 98.11093761,
 97.6386369, 97.127325, 96.57705093, 95.98786772, 95.35983241,
 94.69300606, 93.98745377, 93.24324466, 92.46045187, 91.63915259,
 90.77942802, 89.88136342, 88.94504806, 87.97057526, 86.95804237,
 85.9075508, 84.81920601, 83.69311754, 82.52939897, 81.32816897,
 80.08955029, 78.81366979, 77.50065843, 76.15065131, 74.76378766,
 73.34021084, 71.88006839, 70.383512, 68.85069754, 67.28178507,
 65.67693884, 64.03632733, 62.36012321, 60.64850342, 58.90164912,
 57.11974577, 55.30298307, 53.45155505, 51.56566005, 49.64550071,
 47.69128401, 45.70322127, 43.6815282, 41.6264249, 39.53813589,
 37.41689014, 35.2629211, 33.07646673, 30.85776951, 28.60707648,
 26.32463928, 24.01071415, 21.66556197, 19.2894483, 16.8826434,
 14.44542227, 11.97806469, 9.48085528, 6.95408352, 4.39804377,
 1.81303531, 0., 0., 0., 0.,
 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0.])

D_s_experimental_T = np.array([100., 98.94032162, 98.57597785, 97.72468545, 97.36952287,
 96.66291527, 96.36639459, 95.40392186, 95.1539977, 94.57751099,
 94.22146801, 93.45740784, 93.14750558, 92.61417989, 92.34313954,
 91.85865718, 91.49565868, 91.08263489, 90.78469486, 90.39393719,
 90.01484882, 89.56667679, 89.23636571, 88.85388943, 88.48261692,
 88.06065508, 87.7216227, 87.32940267, 86.96873583, 86.56179563,
 86.23125512, 85.84777127, 85.49012216, 85.09568149, 84.76724141,
 84.39167637, 84.03429163, 83.64543728, 83.31141588, 82.94215815,
 82.58906769, 82.21219578, 81.88352058, 81.5275011, 81.18218313,
 80.81901917, 80.50132829, 80.15982059, 79.82758348, 79.48071873,
 79.18175019, 78.86434509, 78.54965966, 78.22608919, 77.93726146,
 77.63590579, 77.33067875, 77.0322857, 76.75392832, 76.46668539,
 76.17904318, 75.88730533, 75.62418817, 75.35069755, 75.07333259,
 74.79580643, 74.54479454, 74.28478615, 74.02287045, 73.76007138,
 73.52262866, 73.27620927, 73.02871197, 72.78096725, 72.5575413,
 72.3257142, 72.09356017, 71.86187116, 71.65269923, 71.43582852,
 71.21933266, 71.00399612, 70.81078262, 70.61042098, 70.41096667,
 70.2122495, 70.03518254, 69.85142911, 69.66832566, 69.48576598,
 69.32461363, 69.15717292, 68.99031354, 68.82399682, 68.67895558,
 68.52795477, 68.37749655, 68.22756068, 68.09871038, 67.96417733,
 67.83015325, 67.69662238, 67.58401593, 67.46605295, 67.34856556,
 67.23154138, 67.13530796, 67.03398813, 66.93311778, 66.83268594])  # note: len 110, extends beyond T_c

# --- File 3: Current dependence ---------------------------------------------
I_dc = np.array([0., 1.2244898, 2.44897959, 3.67346939, 4.89795918, 6.12244898,
 7.34693878, 8.57142857, 9.79591837, 11.02040816, 12.24489796, 13.46938776,
 14.69387755, 15.91836735, 17.14285714, 18.36734694, 19.59183673, 20.81632653,
 22.04081633, 23.26530612, 24.48979592, 25.71428571, 26.93877551, 28.16326531,
 29.3877551, 30.6122449, 31.83673469, 33.06122449, 34.28571429, 35.51020408,
 36.73469388, 37.95918367, 39.18367347, 40.40816327, 41.63265306, 42.85714286,
 44.08163265, 45.30612245, 46.53061224, 47.75510204, 48.97959184, 50.20408163,
 51.42857143, 52.65306122, 53.87755102, 55.10204082, 56.32653061, 57.55102041,
 58.7755102, 60.])
I_c = 50.0  # critical current (nA)

D_s_gl = np.array([100., 99.9400128, 99.76010238, 99.46047673, 99.04142285,
 98.50330382, 97.8465578, 97.07169701, 96.17930774, 95.17005029,
 94.04465894, 92.80394191, 91.44878132, 89.98013312, 88.39902698,
 86.70656626, 84.9039279, 82.99236236, 81.0731935, 79.24781846,
 77.31770759, 75.48440437, 73.74852547, 71.91076077, 70.07187332,
 68.33269927, 66.49314783, 64.55410134, 62.7165152, 60.88041783,
 59.04581878, 57.11441065, 55.08487616, 53.0577791, 51.03312356,
 49.01090771, 46.89212476, 44.67575996, 42.46288158, 40.15347288,
 37.74751014, 35.24565167, 32.74814983, 30.15504706, 27.56715703,
 24.88428641, 22.10623592, 19.33440022, 16.56778521, 13.7063918,
 10.85121599, 8.00224987, 5.05948264, 2.12289956, 0.,
 0., 0., 0., 0., 0.])

D_s_linear = np.array([100., 97.55102041, 95.10204082, 92.65306122, 90.20408163,
 87.75510204, 85.30612245, 82.85714286, 80.40816327, 77.95918367,
 75.51020408, 73.06122449, 70.6122449, 68.16326531, 65.71428571,
 63.26530612, 60.81632653, 58.36734694, 55.91836735, 53.46938776,
 51.02040816, 48.57142857, 46.12244898, 43.67346939, 41.2244898,
 38.7755102, 36.32653061, 33.87755102, 31.42857143, 28.97959184,
 26.53061224, 24.08163265, 21.63265306, 19.18367347, 16.73469388,
 14.28571429, 11.83673469, 9.3877551, 6.93877551, 4.48979592,
 2.04081633, 0., 0., 0., 0.,
 0., 0., 0., 0., 0.])

D_s_dc_exp = np.array([100., 99.95173681, 99.77122062, 99.46074713, 99.02187532,
 98.45641968, 97.76644627, 96.95426875, 96.02244633, 94.97378175,
 93.81131728, 92.53833265, 91.1583411, 89.67508639, 88.09253877,
 86.41489101, 84.64655341, 82.79215086, 80.85651697, 78.84468821,
 76.76189805, 74.61357116, 72.4053177, 70.14292664, 67.83235928,
 65.4797428, 63.09136199, 60.67365091, 58.2331856, 55.77667381,
 53.31094597, 50.84294515, 48.37971824, 45.92840824, 43.49624357,
 41.09052863, 38.71863646, 36.38799853, 34.10609457, 31.88043854,
 29.71856768, 27.62803166, 25.61638179, 23.69115947, 21.85988681,
 20.13005843, 18.50913233, 17.00452091, 15.6235851, 14.37362461,
 13.2618704, 12.29548021, 11.48153341, 10.82702596, 10.33886858,
 10.02388402, 9.88880416, 9.94026741, 10.18481622, 10.6288957,
 11.27885154, 12.14092878, 13.22126972, 14.52591304, 16.060792,
 17.83173345, 19.84445699, 22.10457433, 24.61758961, 27.38889886,
 30.42378952, 33.72743997, 37.30492021, 41.16129148, 45.30160693,
 49.7309113, 54.45424068, 59.47662226, 64.80307515, 70.43861021,
 76.38822986, 82.65692897, 89.24969482, 96.17150801, 103.42734248])

P_mw = np.array([0., 0.02040816, 0.04081633, 0.06122449, 0.08163265, 0.10204082,
 0.12244898, 0.14285714, 0.16326531, 0.18367347, 0.20408163, 0.2244898,
 0.24489796, 0.26530612, 0.28571429, 0.30612245, 0.32653061, 0.34693878,
 0.36734694, 0.3877551, 0.40816327, 0.42857143, 0.44897959, 0.46938776,
 0.48979592, 0.51020408, 0.53061224, 0.55102041, 0.57142857, 0.59183673,
 0.6122449, 0.63265306, 0.65306122, 0.67346939, 0.69387755, 0.71428571,
 0.73469388, 0.75510204, 0.7755102, 0.79591837, 0.81632653, 0.83673469,
 0.85714286, 0.87755102, 0.89795918, 0.91836735, 0.93877551, 0.95918367,
 0.97959184, 1.])

I_mw = np.array([0., 2.85773803, 4.04081633, 4.94974747, 5.71447606, 6.38806117,
 7., 7.56497728, 8.09284713, 8.59016994, 9.06166058, 9.51103601,
 9.94123006, 10.35460675, 10.75309989, 11.13831958, 11.51164078, 11.8742578,
 12.22721636, 12.57142857, 12.90769231, 13.23671495, 13.55912678, 13.87549008,
 14.18630837, 14.4920399, 14.79310345, 15.08988542, 15.3827453, 15.67201931,
 15.95802331, 16.24105541, 16.52139866, 16.79932309, 17.07508725, 17.34893957,
 17.62112057, 17.89186305, 18.16139318, 18.42993055, 18.69768922, 18.96487772,
 19.23170001, 19.49835537, 19.76503831, 20.03193944, 20.29924533, 20.56713933,
 20.83580146, 21.1054086])

D_s_mw_exp = np.array([100., 99.96555237, 99.88725513, 99.78282278, 99.64829843,
 99.49441822, 99.3222791, 99.13731067, 98.94242671, 98.73958297,
 98.52974471, 98.31369853, 98.09203829, 97.86518943, 97.63344443,
 97.39699898, 97.15597213, 96.91043373, 96.66041868, 96.40594036,
 96.14700201, 95.88360233, 95.61573949, 95.34341343, 95.06662679,
 94.78538512, 94.4996971, 94.20957473, 93.91503335, 93.61609178,
 93.3127724, 93.00510117, 92.69310771, 92.37682529, 92.05629079,
 91.73154466, 91.40263086, 91.06959681, 90.73249335, 90.39137467,
 90.04629821, 89.69732458, 89.34451744, 88.98794346, 88.62767222,
 88.26377613, 87.89633036, 87.52541277, 87.15110388, 86.77348682])

print("Data loaded successfully.")
print(f"  n_eff range: {n_eff[0]:.2e} to {n_eff[-1]:.2e} m^-2")
print(f"  T range: {T[0]:.3f} to {T[-1]:.3f} K  (T_c = {T_c} K)")
print(f"  I_dc range: {I_dc[0]:.1f} to {I_dc[-1]:.1f} nA  (I_c = {I_c} nA)")
print(f"  D_s_exp_hole at n_min: {D_s_exp_hole[0]:.3e}, n_max: {D_s_exp_hole[-1]:.3e}")
print(f"  Ratio exp/geom at max n: {D_s_exp_hole[-1]/D_s_geom[-1]:.2f}x")
print(f"  Ratio exp/conv at max n: {D_s_exp_hole[-1]/D_s_conv[-1]:.2f}x")

# =============================================================================
# FIGURE 1: Carrier-density dependence — quantum geometry enhancement
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot(n_eff*1e-15, D_s_conv*1e-9, 'b-', label='Conventional (Fermi liquid)', lw=2)
ax.plot(n_eff*1e-15, D_s_geom*1e-9, 'g--', label='Quantum geometric prediction', lw=2)
ax.plot(n_eff*1e-15, D_s_exp_hole*1e-9, 'r-o', label='Experiment (hole-doped)',
        markersize=4, markevery=5)
ax.plot(n_eff*1e-15, D_s_exp_electron*1e-9, 'm-s', label='Experiment (electron-doped)',
        markersize=4, markevery=5)
ax.set_xlabel('Carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
ax.set_ylabel('Superfluid stiffness $D_s$ (10$^9$ Hz)')
ax.set_title('(a) Superfluid Stiffness vs Carrier Density')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ratio_hole     = D_s_exp_hole     / D_s_conv
ratio_electron = D_s_exp_electron / D_s_conv
ratio_geom     = D_s_geom         / D_s_conv
ax.plot(n_eff*1e-15, ratio_geom,     'g--', label='Quantum geometric / Conventional', lw=2)
ax.plot(n_eff*1e-15, ratio_hole,     'r-o', label='Exp (hole-doped) / Conventional',
        markersize=4, markevery=5)
ax.plot(n_eff*1e-15, ratio_electron, 'm-s', label='Exp (electron-doped) / Conventional',
        markersize=4, markevery=5)
ax.axhline(1, color='b', lw=1.5, linestyle=':', label='Conventional baseline')
ax.set_xlabel('Carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
ax.set_ylabel('Enhancement ratio $D_s / D_{s,conv}$')
ax.set_title('(b) Enhancement over Conventional Prediction')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig1_carrier_density.png"), bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# =============================================================================
# FIGURE 2: Temperature dependence — power-law analysis
# =============================================================================
# The experimental temperature array has 110 points (T goes to 1.2), whereas
# T has 100 points (to 1.0). Align lengths.
T_exp = T[:len(D_s_experimental_T)]   # same length as experimental array (110 if needed, else 100)
# Actually T has 100 pts, D_s_experimental_T has 110 pts — use the first 100
D_s_exp_T_aligned = D_s_experimental_T[:len(T)]
# Only use T < T_c for fitting
mask = (T > 0) & (T < T_c)
T_fit = T[mask]
D_exp_fit = D_s_exp_T_aligned[mask]

# Fit experimental data to: D_s = D_s0 * (1 - (T/T_c)^n)
def power_law_model(T, D0, n):
    Tc = 1.0
    result = D0 * (1 - (T / Tc)**n)
    return np.where(T < Tc, result, 0.0)

# The experimental data has a large residual (doesn't reach 0 at Tc),
# consistent with a quantum geometry background plus a SC component.
# Use a model with residual: D_s = Dbg + Dsc * (1 - (T/Tc)^n)
def power_law_residual(T, Dbg, Dsc, n):
    Tc = 1.0
    sc_part = np.where(T < Tc, Dsc * (1 - (T/Tc)**n), 0.0)
    return Dbg + sc_part

D_exp_fit = D_s_exp_T_aligned[mask]
popt, pcov = curve_fit(power_law_residual, T_fit, D_exp_fit,
                        p0=[65.0, 35.0, 2.5], bounds=([0, 5, 1.0], [100, 100, 8.0]))
Dbg_fit, Dsc_fit, n_fit = popt
perr = np.sqrt(np.diag(pcov))
print(f"\nPower-law fit (with residual): Dbg = {Dbg_fit:.2f} ± {perr[0]:.2f}, "
      f"Dsc = {Dsc_fit:.2f} ± {perr[1]:.2f}, n = {n_fit:.3f} ± {perr[2]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot(T_for(D_s_bcs),           D_s_bcs,           'b-',  label='BCS (s-wave, n≈2)', lw=2)
ax.plot(T_for(D_s_nodal),         D_s_nodal,         'c--', label='Nodal d-wave (n=1, T-linear)', lw=2)
ax.plot(T_for(D_s_power_n2_5),    D_s_power_n2_5,    'g-',  label='Power law n=2.5', lw=2)
ax.plot(T_for(D_s_power_n3),      D_s_power_n3,      'm-',  label='Power law n=3.0', lw=2)
ax.plot(T, D_s_exp_T_aligned,
        'r-o', label='Experiment', markersize=3, markevery=8)
T_fine = np.linspace(0, T_c * 1.15, 300)
D_fit_curve = power_law_residual(T_fine, Dbg_fit, Dsc_fit, n_fit)
ax.plot(T_fine, D_fit_curve, 'k--', lw=2,
        label=f'Best fit: n={n_fit:.2f}±{perr[2]:.2f}')
ax.axvline(T_c, color='gray', linestyle=':', lw=1.5, label='$T_c$')
ax.set_xlabel('Temperature $T/T_c$')
ax.set_ylabel('Superfluid stiffness $D_s / D_{s0}$ (%)')
ax.set_title('(a) Temperature Dependence of $D_s$')
ax.legend(fontsize=9)
ax.set_xlim(0, 1.1)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# Log-log plot: show D_s suppression = D_sc component only (subtract background)
ax = axes[1]
T_nz = T_fit
# Normalize to remove background: only the SC-driven suppression
Dsc_data = D_exp_fit - Dbg_fit   # superconducting component
valid = Dsc_data > 0.1
ax.loglog(T_nz[valid], Dsc_data[valid], 'ro', markersize=4,
          label='Experiment SC component')
for n_val, color, label in [(2.0,'b','n=2 (BCS)'), (2.5,'g','n=2.5'), (3.0,'m','n=3')]:
    dD = Dsc_fit * (1 - (T_nz/T_c)**n_val)
    valid_th = dD > 0.01
    ax.loglog(T_nz[valid_th], dD[valid_th], color=color, linestyle='--', label=label, lw=1.5)
dD_fit = power_law_residual(T_nz, Dbg_fit, Dsc_fit, n_fit) - Dbg_fit
valid_fit = dD_fit > 0.01
ax.loglog(T_nz[valid_fit], dD_fit[valid_fit], 'k-', lw=2, label=f'Best fit n={n_fit:.2f}')
ax.set_xlabel('Temperature $T/T_c$')
ax.set_ylabel('$D_{s0} - D_s(T)$ (arb. units)')
ax.set_title('(b) Log-Log: Power-Law Scaling')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig2_temperature.png"), bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# =============================================================================
# FIGURE 3: Current dependence — GL vs Meissner vs experiment
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Pre-computed model/exp arrays may differ in size from I_dc (50 pts).
# Build proper independent I-axes from array lengths.
I_gl_axis   = np.linspace(0, I_dc[-1], len(D_s_gl))
I_lin_axis  = np.linspace(0, I_dc[-1], len(D_s_linear))
I_dc_exp_ax = np.linspace(0, I_dc[-1], len(D_s_dc_exp))

ax.plot(I_gl_axis,   D_s_gl,    'b-',  label='Ginzburg-Landau (I² dep.)', lw=2)
ax.plot(I_lin_axis,  D_s_linear,'g--', label='Linear Meissner model', lw=2)
ax.plot(I_dc_exp_ax, D_s_dc_exp,'r-o', markersize=4, markevery=8,
        label='Experiment (DC bias)')
ax.axvline(I_c, color='gray', linestyle=':', lw=1.5, label=f'$I_c$ = {I_c} nA')
ax.set_xlabel('DC Current $I_{dc}$ (nA)')
ax.set_ylabel('Superfluid stiffness $D_s / D_{s0}$ (%)')
ax.set_title('(a) DC Current Dependence of $D_s$')
ax.legend(fontsize=9)
ax.set_xlim(0, 65)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(I_mw, D_s_mw_exp, 'r-o', markersize=4, label='Experiment (microwave)')
# Fit GL model to microwave data (quadratic suppression)
def gl_model(I, Ds0, Ic):
    x = I / Ic
    result = Ds0 * (1 - x**2)
    return np.where(x < 1, result, 0.0)

popt_mw, pcov_mw = curve_fit(gl_model, I_mw, D_s_mw_exp, p0=[100.0, 25.0])
Ds0_mw, Ic_mw = popt_mw
I_fine = np.linspace(0, I_mw[-1]*1.05, 200)
ax.plot(I_fine, gl_model(I_fine, Ds0_mw, Ic_mw), 'b--', lw=2,
        label=f'GL fit: $I_c$={Ic_mw:.1f} nA')
ax.set_xlabel('Microwave current amplitude $I_{mw}$ (nA)')
ax.set_ylabel('Superfluid stiffness $D_s / D_{s0}$ (%)')
ax.set_title('(b) Microwave Current Dependence of $D_s$')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig3_current.png"), bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# =============================================================================
# FIGURE 4: Summary / overview panel
# =============================================================================
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.38)

# (a) Stiffness vs n — absolute
ax1 = fig.add_subplot(gs[0, :2])
ax1.fill_between(n_eff*1e-15, D_s_conv*1e-9, D_s_geom*1e-9, alpha=0.15, color='green',
                  label='Quantum geometry contribution')
ax1.fill_between(n_eff*1e-15, D_s_geom*1e-9, D_s_exp_hole*1e-9, alpha=0.15, color='red',
                  label='Residual (beyond geom.)')
ax1.plot(n_eff*1e-15, D_s_conv*1e-9,      'b-',  lw=2.0, label='Conventional')
ax1.plot(n_eff*1e-15, D_s_geom*1e-9,      'g--', lw=2.0, label='+ Quantum geometry')
ax1.plot(n_eff*1e-15, D_s_exp_hole*1e-9,  'r-',  lw=2.0, label='Experiment (hole)')
ax1.plot(n_eff*1e-15, D_s_exp_electron*1e-9, 'm:', lw=2.0, label='Experiment (electron)')
ax1.set_xlabel('$n_{eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax1.set_ylabel('$D_s$ (10$^9$ Hz)', fontsize=11)
ax1.set_title('(a) Superfluid Stiffness Enhancement by Quantum Geometry')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# (b) Ratio
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(n_eff*1e-15, ratio_hole,     'r-', lw=2, label='Hole-doped / Conv')
ax2.plot(n_eff*1e-15, ratio_electron, 'm--', lw=2, label='Electron-doped / Conv')
ax2.plot(n_eff*1e-15, ratio_geom,     'g:', lw=2, label='Geom. / Conv')
ax2.axhline(1, color='b', lw=1.0, linestyle=':')
ax2.set_xlabel('$n_{eff}$ ($10^{15}$ m$^{-2}$)', fontsize=11)
ax2.set_ylabel('Enhancement ratio', fontsize=11)
ax2.set_title('(b) Enhancement ratio')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# (c) Temperature
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(T_for(D_s_bcs),        D_s_bcs,        'b-',  lw=1.5, label='BCS (n≈2)')
ax3.plot(T_for(D_s_nodal),      D_s_nodal,      'c--', lw=1.5, label='d-wave (n=1)')
ax3.plot(T_for(D_s_power_n2_5), D_s_power_n2_5, 'g-',  lw=1.5, label='n=2.5')
ax3.plot(T_for(D_s_power_n3),   D_s_power_n3,   'm-',  lw=1.5, label='n=3')
ax3.plot(T, D_s_exp_T_aligned,
         'r-', lw=2, label='Experiment')
T_fine2 = np.linspace(0, T_c*1.15, 300)
ax3.plot(T_fine2, power_law_residual(T_fine2, Dbg_fit, Dsc_fit, n_fit), 'k--', lw=1.5,
         label=f'Fit n={n_fit:.2f}')
ax3.axvline(T_c, color='gray', lw=1, linestyle=':')
ax3.set_xlabel('$T$ (K) [$T_c=1$ K]', fontsize=11)
ax3.set_ylabel('$D_s$ (%)', fontsize=11)
ax3.set_title('(c) Temp. Dependence')
ax3.legend(fontsize=7)
ax3.set_xlim(0, 1.05)
ax3.grid(True, alpha=0.3)

# (d) DC current
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(I_gl_axis,   D_s_gl,    'b-',  lw=2, label='GL (I²)')
ax4.plot(I_lin_axis,  D_s_linear,'g--', lw=2, label='Meissner (I)')
ax4.plot(I_dc_exp_ax, D_s_dc_exp,'r-o', markersize=3, markevery=8,
         label='Experiment')
ax4.axvline(I_c, color='gray', lw=1, linestyle=':')
ax4.set_xlabel('$I_{dc}$ (nA)', fontsize=11)
ax4.set_ylabel('$D_s$ (%)', fontsize=11)
ax4.set_title('(d) DC Current Dep.')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# (e) MW current
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(I_mw, D_s_mw_exp, 'r-o', markersize=3, label='Exp. (microwave)')
ax5.plot(I_fine, gl_model(I_fine, Ds0_mw, Ic_mw), 'b--', lw=2,
         label=f'GL fit $I_c$={Ic_mw:.1f} nA')
ax5.set_xlabel('$I_{mw}$ (nA)', fontsize=11)
ax5.set_ylabel('$D_s$ (%)', fontsize=11)
ax5.set_title('(e) Microwave Current Dep.')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

plt.suptitle('MATBG Superfluid Stiffness — Summary Overview', fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(IMG_DIR, "fig4_summary.png"), bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# =============================================================================
# FIGURE 5: Deeper analysis — quadratic I² scaling check
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# Check if D_s(I) ~ 1 - (I/Ic)^2 for DC data (use proper I axis)
mask_below_Ic = I_dc_exp_ax <= I_c
I_dc_sub     = I_dc_exp_ax[mask_below_Ic]
D_dc_sub     = D_s_dc_exp[mask_below_Ic]
suppression = 1 - D_dc_sub / 100.0
I2 = (I_dc_sub / I_c)**2
mask_nonzero = suppression > 0.001
ax.plot(I2[mask_nonzero], suppression[mask_nonzero], 'ro', markersize=4, label='Experiment')
pfit = np.polyfit(I2[mask_nonzero], suppression[mask_nonzero], 1)
ax.plot(I2[mask_nonzero], np.polyval(pfit, I2[mask_nonzero]), 'b--', lw=2,
        label=f'Linear fit: slope={pfit[0]:.3f}')
ax.set_xlabel('$(I_{dc}/I_c)^2$', fontsize=12)
ax.set_ylabel('$1 - D_s/D_{s0}$', fontsize=12)
ax.set_title('(a) Quadratic (I²) Suppression Check — DC')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
# Same for microwave
suppression_mw = 1 - D_s_mw_exp / 100.0
I2_mw = (I_mw / Ic_mw)**2
mask_mw = suppression_mw > 0.001
ax.plot(I2_mw[mask_mw], suppression_mw[mask_mw], 'ro', markersize=4, label='Experiment (MW)')
pfit_mw = np.polyfit(I2_mw[mask_mw], suppression_mw[mask_mw], 1)
ax.plot(I2_mw[mask_mw], np.polyval(pfit_mw, I2_mw[mask_mw]), 'b--', lw=2,
        label=f'Linear fit: slope={pfit_mw[0]:.3f}')
ax.set_xlabel('$(I_{mw}/I_c)^2$', fontsize=12)
ax.set_ylabel('$1 - D_s/D_{s0}$', fontsize=12)
ax.set_title('(b) Quadratic (I²) Suppression Check — Microwave')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig5_quadratic_scaling.png"), bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# =============================================================================
# NUMERICAL SUMMARY
# =============================================================================
print("\n========== NUMERICAL SUMMARY ==========")
print(f"\n--- Quantum Geometry Enhancement ---")
mean_ratio_hole     = np.mean(ratio_hole)
mean_ratio_electron = np.mean(ratio_electron)
mean_ratio_geom     = np.mean(ratio_geom)
print(f"Mean D_s(exp,hole) / D_s(conv)     = {mean_ratio_hole:.1f}x")
print(f"Mean D_s(exp,elec) / D_s(conv)     = {mean_ratio_electron:.1f}x")
print(f"Mean D_s(geom)     / D_s(conv)     = {mean_ratio_geom:.1f}x")
print(f"Ratio at n=5e15 m^-2 (hole/geom)   = {D_s_exp_hole[-1]/D_s_geom[-1]:.2f}x")

print(f"\n--- Temperature Dependence Power-Law Fit ---")
print(f"  Background D_bg = {Dbg_fit:.2f} ± {perr[0]:.2f} (quantum geometry residual)")
print(f"  SC component   = {Dsc_fit:.2f} ± {perr[1]:.2f}")
print(f"  n    = {n_fit:.3f} ± {perr[2]:.3f}  (BCS~2, d-wave~1, MATBG~2.5-3)")

print(f"\n--- Current Dependence GL Fit (microwave) ---")
print(f"  D_s0 = {Ds0_mw:.2f}")
print(f"  I_c  = {Ic_mw:.2f} nA  (input I_c = {I_c} nA)")
print(f"  DC quadratic suppression slope: {pfit[0]:.4f}")
print(f"  MW quadratic suppression slope: {pfit_mw[0]:.4f}")

# Save numerical outputs
import json
results = {
    "quantum_geometry_enhancement": {
        "mean_ratio_hole_doped": float(mean_ratio_hole),
        "mean_ratio_electron_doped": float(mean_ratio_electron),
        "mean_ratio_geometric": float(mean_ratio_geom),
        "max_exp_to_geom_ratio": float(D_s_exp_hole[-1]/D_s_geom[-1]),
    },
    "temperature_power_law": {
        "D_bg_fit": float(Dbg_fit),
        "D_bg_err": float(perr[0]),
        "D_sc_fit": float(Dsc_fit),
        "D_sc_err": float(perr[1]),
        "power_law_exponent_n": float(n_fit),
        "power_law_exponent_err": float(perr[2]),
    },
    "current_dependence": {
        "GL_fit_Ds0_mw": float(Ds0_mw),
        "GL_fit_Ic_mw_nA": float(Ic_mw),
        "DC_quadratic_slope": float(pfit[0]),
        "MW_quadratic_slope": float(pfit_mw[0]),
    }
}
with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR}/results_summary.json")
print("Analysis complete.")
