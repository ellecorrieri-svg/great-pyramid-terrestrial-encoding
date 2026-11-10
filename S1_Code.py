"""
S1 Code — Reproduction of the Thuban and Sirius worked stellar applications
for "A continuous celestial coordinate map from Great Pyramid geometry".

This script reproduces the astrometric propagation to EPJ = -2599.0
(approximately 2600 BCE) and the numerical Great Pyramid transformations.

Dependencies: Python 3 standard library only.

Sources
-------
SIMBAD astrometric inputs:
https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Alpha+Draconis
https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Sirius

Long-term precession:
Vondrak, Capitaine & Wallace (2011), A&A 534, A22,
doi:10.1051/0004-6361/201117274, with the corrected coefficient
reported in the 2012 corrigendum.

Nutation, atmospheric refraction, and short-period apparent-place effects
are intentionally excluded, matching the idealized examples in the manuscript.
"""

from math import sin, cos, sqrt, atan2, asin, atan, tan, radians, degrees, pi

TAU = 2.0 * pi
AS2R = pi / (180.0 * 3600.0)
EPS0 = 84381.406 * AS2R
AU_KM = 149597870.7
SEC_PER_JULIAN_YEAR = 365.25 * 86400.0
KM_S_TO_AU_YR = SEC_PER_JULIAN_YEAR / AU_KM
PC_TO_AU = 648000.0 / pi

TARGET_EPJ = -2599.0
PHI_DEG = 29.979182
H_CUBIT = 280.0
A_CUBIT = 220.0
H_METRE = 146.6
CUBIT_METRE = H_METRE / H_CUBIT
EPSILON_DEG = 23.983956
SIRIUS_HOUR_ANGLE_DEG = 30.0

PQ_POLY = (
    (5851.607687, -1600.886300),
    (-0.1189000, 1.1689818),
    (-0.00028913, -0.00000020),
    (0.000000101, -0.000000437),
)
PQ_PERIODIC = (
    (708.15, -5486.751211, -684.661560, 667.666730, -5523.863691),
    (2309.00, -17.127623, 2446.283880, -2354.886252, -549.747450),
    (1620.00, -617.517403, 399.671049, -428.152441, -310.998056),
    (492.20, 413.442940, -356.652376, 376.202861, 421.535876),
    (1183.00, 78.614193, -186.387003, 184.778874, -36.776172),
    (622.00, -180.732815, -316.800070, 335.321713, -145.278396),
    (882.00, -87.676083, 198.296701, -185.138669, -34.744450),
    (547.00, 46.140315, 101.135679, -120.972830, 22.885731),
)
XY_POLY = (
    (5453.282155, -73750.930350),
    (0.4252841, -0.7675452),
    (-0.00037173, -0.00018725),
    (-0.000000152, 0.000000231),
)
XY_PERIODIC = (
    (256.75, -819.940624, 75004.344875, 81491.287984, 1558.515853),
    (708.15, -8444.676815, 624.033993, 787.163481, 7774.939698),
    (274.20, 2600.009459, 1251.136893, 1251.296102, -2219.534038),
    (241.45, 2755.175630, -1102.212834, -1257.950837, -2523.969396),
    (2309.00, -167.659835, -2660.664980, -2966.799730, 247.850422),
    (492.20, 871.855056, 699.291817, 639.744522, -846.485643),
    (396.10, 44.769698, 153.167220, 131.600209, -1393.124055),
    (288.90, -512.313065, -950.865637, -445.040117, 368.526116),
    (231.10, -819.415595, 499.754645, 584.522874, 749.045012),
    (1610.00, -538.071099, -145.188210, -89.756563, 444.704518),
    (620.00, -189.793622, 558.116553, 524.429630, 235.934465),
    (157.87, -402.922932, -23.923029, -13.549067, 374.049623),
    (220.30, 179.516345, -165.405086, -210.157124, -171.330180),
    (1200.00, -9.814756, 9.344131, -44.919798, -22.899655),
)

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def norm(v):
    return sqrt(dot(v, v))

def unit(v):
    n = norm(v)
    return tuple(x/n for x in v)

def mat_vec(m, v):
    return tuple(dot(row, v) for row in m)

def mat_mul(a, b):
    bt = tuple(zip(*b))
    return tuple(tuple(dot(row, col) for col in bt) for row in a)

def hms_to_deg(h, m, s):
    return 15.0 * (h + m/60.0 + s/3600.0)

def dms_to_deg(sign, d, m, s):
    value = d + m/60.0 + s/3600.0
    return value if sign >= 0 else -value

def long_term_poles(epj):
    T = (epj - 2000.0) / 100.0

    P = Q = 0.0
    for period, pcos, qcos, psin, qsin in PQ_PERIODIC:
        angle = TAU * T / period
        P += cos(angle)*pcos + sin(angle)*psin
        Q += cos(angle)*qcos + sin(angle)*qsin
    power = 1.0
    for p, q in PQ_POLY:
        P += p*power
        Q += q*power
        power *= T
    P *= AS2R
    Q *= AS2R
    z = sqrt(max(0.0, 1.0 - P*P - Q*Q))
    s0, c0 = sin(EPS0), cos(EPS0)
    ecl_pole = (P, -Q*c0 - z*s0, -Q*s0 + z*c0)

    X = Y = 0.0
    for period, xcos, ycos, xsin, ysin in XY_PERIODIC:
        angle = TAU * T / period
        X += cos(angle)*xcos + sin(angle)*xsin
        Y += cos(angle)*ycos + sin(angle)*ysin
    power = 1.0
    for x, y in XY_POLY:
        X += x*power
        Y += y*power
        power *= T
    X *= AS2R
    Y *= AS2R
    eq_pole = (X, Y, sqrt(max(0.0, 1.0 - X*X - Y*Y)))
    return ecl_pole, eq_pole

def precession_bias_matrix(epj):
    ecl_pole, eq_pole = long_term_poles(epj)
    equinox = unit(cross(eq_pole, ecl_pole))
    middle = cross(eq_pole, equinox)
    precession = (equinox, middle, eq_pole)

    dx = -0.016617 * AS2R
    de = -0.0068192 * AS2R
    dr = -0.0146 * AS2R
    bias = (
        (1.0, dr, -dx),
        (-dr, 1.0, -de),
        (dx, de, 1.0),
    )
    return mat_mul(precession, bias)

def propagate_icrs(ra_deg, dec_deg, pmra_masyr, pmdec_masyr,
                   parallax_mas, radial_velocity_kms, target_epj):
    ra, dec = radians(ra_deg), radians(dec_deg)
    u = (cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec))
    e_ra = (-sin(ra), cos(ra), 0.0)
    e_dec = (-sin(dec)*cos(ra), -sin(dec)*sin(ra), cos(dec))

    distance_au = (1000.0/parallax_mas) * PC_TO_AU
    mas_to_rad = AS2R / 1000.0
    v_tan = tuple(
        distance_au*mas_to_rad*(pmra_masyr*e_ra[i] + pmdec_masyr*e_dec[i])
        for i in range(3)
    )
    v_rad = tuple(radial_velocity_kms*KM_S_TO_AU_YR*u[i] for i in range(3))
    r0 = tuple(distance_au*x for x in u)
    dt = target_epj - 2000.0
    r = tuple(r0[i] + (v_tan[i] + v_rad[i])*dt for i in range(3))

    direction_date = unit(mat_vec(precession_bias_matrix(target_epj), unit(r)))
    ra_date = degrees(atan2(direction_date[1], direction_date[0])) % 360.0
    dec_date = degrees(atan2(direction_date[2],
                             sqrt(direction_date[0]**2 + direction_date[1]**2)))
    return ra_date, dec_date

def cot_deg(angle_deg):
    return 1.0 / tan(radians(angle_deg))

def thuban_application(delta_deg):
    hU = 90.0 + PHI_DEG - delta_deg
    hL = PHI_DEG + delta_deg - 90.0
    RU = H_CUBIT*cot_deg(hU)
    RL = H_CUBIT*cot_deg(hL)
    dU, dL = RU-A_CUBIT, RL-A_CUBIT
    Rstar = H_CUBIT*cot_deg(PHI_DEG)
    sensitivity = abs(-H_CUBIT / ((A_CUBIT+dL)**2 + H_CUBIT**2))*180.0/pi*60.0
    return {
        "h_upper_deg": hU, "h_lower_deg": hL,
        "R_upper_cubit": RU, "R_lower_cubit": RL,
        "d_upper_cubit": dU, "d_lower_cubit": dL,
        "d_upper_m": dU*CUBIT_METRE, "d_lower_m": dL*CUBIT_METRE,
        "R_star_cubit": Rstar, "R_star_m": Rstar*CUBIT_METRE,
        "phi_recovered_deg": (hU+hL)/2.0,
        "delta_recovered_deg": 90.0-(hU-hL)/2.0,
        "sensitivity_arcmin_per_cubit": sensitivity,
        "cubit_per_arcmin": 1.0/sensitivity,
        "m_per_arcmin": CUBIT_METRE/sensitivity,
        "m_per_10arcmin": 10.0*CUBIT_METRE/sensitivity,
    }

def horizontal_from_equatorial(delta_deg, hour_angle_deg):
    phi, dec, Hh = radians(PHI_DEG), radians(delta_deg), radians(hour_angle_deg)
    h = asin(sin(phi)*sin(dec) + cos(phi)*cos(dec)*cos(Hh))
    sinA = -cos(dec)*sin(Hh)/cos(h)
    cosA = (sin(dec)-sin(phi)*sin(h))/(cos(phi)*cos(h))
    return degrees(h), degrees(atan2(sinA, cosA)) % 360.0

def equatorial_from_horizontal(A_deg, h_deg, lst_deg):
    phi, A, h = radians(PHI_DEG), radians(A_deg), radians(h_deg)
    dec = asin(sin(phi)*sin(h) + cos(phi)*cos(h)*cos(A))
    sinH = -cos(h)*sin(A)/cos(dec)
    cosH = (sin(h)-sin(phi)*sin(dec))/(cos(phi)*cos(dec))
    Hh = atan2(sinH, cosH)
    alpha = (radians(lst_deg)-Hh) % TAU
    return degrees(alpha), degrees(dec), degrees(Hh) % 360.0

def ecliptic_from_equatorial(alpha_deg, delta_deg):
    a, d, e = radians(alpha_deg), radians(delta_deg), radians(EPSILON_DEG)
    x = cos(d)*cos(a)
    y = cos(d)*sin(a)
    z = sin(d)
    xe = x
    ye = y*cos(e) + z*sin(e)
    ze = -y*sin(e) + z*cos(e)
    return degrees(atan2(ye, xe)) % 360.0, degrees(asin(ze))

def sirius_application(alpha_deg, delta_deg):
    lst = (alpha_deg + SIRIUS_HOUR_ANGLE_DEG) % 360.0
    h, A = horizontal_from_equatorial(delta_deg, SIRIUS_HOUR_ANGLE_DEG)
    R = H_CUBIT*cot_deg(h)
    theta = (A-180.0) % 360.0
    x = R*sin(radians(theta))
    y = R*cos(radians(theta))

    h2 = degrees(atan(H_CUBIT/R))
    A2 = (theta+180.0) % 360.0
    alpha2, delta2, H2 = equatorial_from_horizontal(A2, h2, lst)
    lam, beta = ecliptic_from_equatorial(alpha2, delta2)
    return {
        "LST_deg": lst, "h_deg": h, "A_deg": A,
        "R_cubit": R, "R_m": R*CUBIT_METRE,
        "theta_deg": theta,
        "x_cubit": x, "y_cubit": y,
        "x_m": x*CUBIT_METRE, "y_m": y*CUBIT_METRE,
        "h_recovered_deg": h2, "A_recovered_deg": A2,
        "hour_angle_recovered_deg": H2,
        "alpha_recovered_deg": alpha2, "delta_recovered_deg": delta2,
        "lambda_ecl_deg": lam, "beta_ecl_deg": beta,
    }

def main():
    thuban_ra0 = hms_to_deg(14, 4, 23.3575968507)
    thuban_dec0 = dms_to_deg(+1, 64, 22, 33.130623243)
    sirius_ra0 = hms_to_deg(6, 45, 8.91728)
    sirius_dec0 = dms_to_deg(-1, 16, 42, 58.0171)

    th_ra, th_dec = propagate_icrs(
        thuban_ra0, thuban_dec0, -56.351, 12.329, 12.5156, -6.1, TARGET_EPJ
    )
    si_ra, si_dec = propagate_icrs(
        sirius_ra0, sirius_dec0, -546.01, -1223.07, 379.21, -5.50, TARGET_EPJ
    )

    print(f"Thuban EPJ -2599: RA={th_ra:.12f} deg, Dec={th_dec:.12f} deg")
    print(f"Sirius EPJ -2599: RA={si_ra:.12f} deg, Dec={si_dec:.12f} deg")

    print("\nThuban application")
    for key, value in thuban_application(th_dec).items():
        print(f"{key}={value:.12f}")

    print("\nSirius application")
    for key, value in sirius_application(si_ra, si_dec).items():
        print(f"{key}={value:.12f}")

if __name__ == "__main__":
    main()
