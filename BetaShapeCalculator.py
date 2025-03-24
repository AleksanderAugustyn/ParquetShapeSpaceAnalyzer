import numpy as np
import pandas as pd
from scipy.special import sph_harm_y

r0: float = 1.16  # fm


def calculate_sphere_volume(number_of_protons: int, number_of_neutrons: int) -> float:
    """Calculate the volume of a spherical nucleus."""

    number_of_nucleons: int = number_of_protons + number_of_neutrons
    return 4 / 3 * np.pi * number_of_nucleons * r0 ** 3


def calculate_volume(number_of_protons: int, number_of_neutrons: int, beta_values: np.ndarray) -> float:
    """Calculate the volume of a deformed nucleus using analytical formula."""
    beta_params = tuple(beta_values)
    number_of_nucleons: int = number_of_protons + number_of_neutrons

    beta10, beta20, beta30, beta40, beta50, beta60, beta70, beta80, beta90, beta100, beta110, beta120 = beta_params

    volume = (number_of_nucleons * r0 ** 3 * (74207381348100 * np.pi ** 1.5 + 648269351730 * np.sqrt(21) * beta100 ** 3 + 5807534192460 * np.sqrt(
        69) * beta10 * beta110 * beta120 + 8429951570040 * beta110 ** 2 * beta120 + 2724132411000 * beta120 ** 3 + 11131107202215 * np.sqrt(5) * beta10 ** 2 * beta20 + 6996695955678 * np.sqrt(
        5) * beta110 ** 2 * beta20 + 6990550416850 * np.sqrt(5) * beta120 ** 2 * beta20 + 2650263619575 * np.sqrt(5) * beta20 ** 3 + 2197030131010 * np.sqrt(161) * beta110 * beta120 * beta30 + 4770474515235 * np.sqrt(
        105) * beta10 * beta20 * beta30 + 7420738134810 * np.sqrt(
        5) * beta20 * beta30 ** 2 + 11968032555765 * beta110 ** 2 * beta40 + 11932146401175 * beta120 ** 2 * beta40 + 23852372576175 * beta20 ** 2 * beta40 + 10601054478300 * np.sqrt(
        21) * beta10 * beta30 * beta40 + 15178782548475 * beta30 ** 2 * beta40 + 7227991689750 * np.sqrt(5) * beta20 * beta40 ** 2 + 4503594822075 * beta40 ** 3 + 1395572678500 * np.sqrt(
        253) * beta110 * beta120 * beta50 + 2409330563250 * np.sqrt(385) * beta20 * beta30 * beta50 + 8432656971375 * np.sqrt(33) * beta10 * beta40 * beta50 + 3335996164500 * np.sqrt(
        77) * beta30 * beta40 * beta50 + 7135325129625 * np.sqrt(
        5) * beta20 * beta50 ** 2 + 12843585233325 * beta40 * beta50 ** 2 + 2832191612250 * np.sqrt(13) * beta110 ** 2 * beta60 + 2813654593750 * np.sqrt(13) * beta120 ** 2 * beta60 + 6486659208750 * np.sqrt(
        13) * beta30 ** 2 * beta60 + 5837993287875 * np.sqrt(65) * beta20 * beta40 * beta60 + 3891995525250 * np.sqrt(13) * beta40 ** 2 * beta60 + 2335197315150 * np.sqrt(429) * beta10 * beta50 * beta60 + 798726124350 * np.sqrt(
        3289) * beta110 * beta50 * beta60 + 908132289225 * np.sqrt(1001) * beta30 * beta50 * beta60 + 3357800061000 * np.sqrt(13) * beta50 ** 2 * beta60 + 22843567156410 * beta120 * beta60 ** 2 + 7083431855955 * np.sqrt(
        5) * beta20 * beta60 ** 2 + 12500173863450 * beta40 * beta60 ** 2 + 1044291885000 * np.sqrt(13) * beta60 ** 3 + 1042707290625 * np.sqrt(345) * beta110 * beta120 * beta70 + 2472247527750 * np.sqrt(
        345) * beta110 * beta40 * beta70 + 4540661446125 * np.sqrt(105) * beta30 * beta40 * beta70 + 3560036439960 * np.sqrt(165) * beta120 * beta50 * beta70 + 8173190603025 * np.sqrt(
        33) * beta20 * beta50 * beta70 + 2136781857000 * np.sqrt(165) * beta40 * beta50 * beta70 + 5993673108885 * np.sqrt(65) * beta10 * beta60 * beta70 + 383388539688 * np.sqrt(
        4485) * beta110 * beta60 * beta70 + 769241468520 * np.sqrt(
        1365) * beta30 * beta60 * beta70 + 506079913500 * np.sqrt(2145) * beta50 * beta60 * beta70 + 12690870642450 * beta120 * beta70 ** 2 + 7051380128100 * np.sqrt(
        5) * beta20 * beta70 ** 2 + 12297741898050 * beta40 * beta70 ** 2 + 3012380437500 * np.sqrt(13) * beta60 * beta70 ** 2 + 2238344983875 * np.sqrt(17) * beta110 ** 2 * beta80 + 2211803343750 * np.sqrt(
        17) * beta120 ** 2 * beta80 + 882945545625 * np.sqrt(2737) * beta110 * beta30 * beta80 + 11125113874875 * np.sqrt(17) * beta120 * beta40 * beta80 + 5609052374625 * np.sqrt(17) * beta40 ** 2 * beta80 + 395559604440 * np.sqrt(
        4301) * beta110 * beta50 * beta80 + 1282069114200 * np.sqrt(1309) * beta30 * beta50 * beta80 + 3247346111625 * np.sqrt(17) * beta50 ** 2 * beta80 + 1714091619240 * np.sqrt(
        221) * beta120 * beta60 * beta80 + 1410276025620 * np.sqrt(
        1105) * beta20 * beta60 * beta80 + 1821887688600 * np.sqrt(221) * beta40 * beta60 * beta80 + 2741266198125 * np.sqrt(17) * beta60 ** 2 * beta80 + 5238168095160 * np.sqrt(85) * beta10 * beta70 * beta80 + 276891723108 * np.sqrt(
        5865) * beta110 * beta70 * beta80 + 668025485820 * np.sqrt(1785) * beta30 * beta70 * beta80 + 433782783000 * np.sqrt(2805) * beta50 * beta70 * beta80 + 2521231453125 * np.sqrt(
        17) * beta70 ** 2 * beta80 + 10415266251390 * beta120 * beta80 ** 2 + 7030172969820 * np.sqrt(5) * beta20 * beta80 ** 2 + 12167607063150 * beta40 * beta80 ** 2 + 2939035522500 * np.sqrt(
        13) * beta60 * beta80 ** 2 + 800070781125 * np.sqrt(17) * beta80 ** 3 + 6111105 * beta100 ** 2 * (
                                                      1440600 * beta120 + 31 * (36975 * np.sqrt(5) * beta20 + 63423 * beta40 + 15080 * np.sqrt(13) * beta60 + 12005 * np.sqrt(17) * beta80)) + 849332484000 * np.sqrt(
        437) * beta110 * beta120 * beta90 + 1000671618375 * np.sqrt(2185) * beta110 * beta20 * beta90 + 4002686473500 * np.sqrt(133) * beta120 * beta30 * beta90 + 1271441585700 * np.sqrt(
        437) * beta110 * beta40 * beta90 + 1785512103375 * np.sqrt(209) * beta120 * beta50 * beta90 + 3188303455050 * np.sqrt(209) * beta40 * beta50 * beta90 + 285681936540 * np.sqrt(
        5681) * beta110 * beta60 * beta90 + 1113375809700 * np.sqrt(1729) * beta30 * beta60 * beta90 + 506079913500 * np.sqrt(2717) * beta50 * beta60 * beta90 + 1241238758760 * np.sqrt(
        285) * beta120 * beta70 * beta90 + 6203093796900 * np.sqrt(57) * beta20 * beta70 * beta90 + 1590536871000 * np.sqrt(285) * beta40 * beta70 * beta90 + 363057329250 * np.sqrt(
        3705) * beta60 * beta70 * beta90 + 1550773449225 * np.sqrt(
        969) * beta10 * beta80 * beta90 + 222786443880 * np.sqrt(7429) * beta110 * beta80 * beta90 + 590770837800 * np.sqrt(2261) * beta30 * beta80 * beta90 + 380345773500 * np.sqrt(
        3553) * beta50 * beta80 * beta90 + 290445863400 * np.sqrt(
        4845) * beta70 * beta80 * beta90 + 9387573945750 * beta120 * beta90 ** 2 + 7015403698875 * np.sqrt(5) * beta20 * beta90 ** 2 + 12078695064150 * beta40 * beta90 ** 2 + 2890627878600 * np.sqrt(
        13) * beta60 * beta90 ** 2 + 2324911563975 * np.sqrt(17) * beta80 * beta90 ** 2 + 55655536011075 * np.sqrt(np.pi) * (beta10 ** 2 + beta100 ** 2 + beta110 ** 2 + beta120 ** 2 + beta20 ** 2 + beta30 ** 2 + beta40 ** 2 +
                                                                                                                             beta50 ** 2 + beta60 ** 2 + beta70 ** 2 + beta80 ** 2 + beta90 ** 2) + 2035 * beta100 * (
                                                      930397104 * np.sqrt(21) * beta110 ** 2 + 912234960 * np.sqrt(21) * beta120 ** 2 + 2242291194 * np.sqrt(105) * beta120 * beta20 + 2841109700 *
                                                      np.sqrt(21) * beta120 * beta40 + 2462010390 * np.sqrt(21) * beta50 ** 2 + 635359725 * np.sqrt(273) * beta120 * beta60 + 1367783550 * np.sqrt(273) * beta40 * beta60 + 1391571090 *
                                                      np.sqrt(21) * beta60 ** 2 + 10160677800 * np.sqrt(5) * beta30 * beta70 + 654157350 * np.sqrt(385) * beta50 * beta70 + 1156074444 * np.sqrt(21) * beta70 ** 2 + 491891400 *
                                                      np.sqrt(357) * beta120 * beta80 + 544322025 * np.sqrt(1785) * beta20 * beta80 + 694207800 * np.sqrt(357) * beta40 * beta80 + 156997764 * np.sqrt(4641) * beta60 * beta80 + 1051409268 *
                                                      np.sqrt(21) * beta80 ** 2 + 1822295475 * np.sqrt(57) * beta30 * beta90 + 166609872 * np.sqrt(4389) * beta50 * beta90 + 377957580 * np.sqrt(665) * beta70 * beta90 + 992760678 *
                                                      np.sqrt(21) * beta90 ** 2 + 8940555 * beta10 * (209 * np.sqrt(161) * beta110 + 230 * np.sqrt(133) * beta90) + 858 * beta110 * (
                                                              1925658 * np.sqrt(69) * beta30 + 175305 * np.sqrt(5313) * beta50 + 394940 * np.sqrt(805) * beta70 + 108045 * np.sqrt(9177) * beta90)))) / (
                     5.5655536011075e13 * np.sqrt(np.pi))

    return volume


def calculate_volume_fixing_factor(number_of_protons: int, number_of_neutrons: int, beta_values: np.ndarray) -> float:
    """Calculate volume fixing factor to conserve volume."""
    initial_volume = calculate_volume(number_of_protons, number_of_neutrons, beta_values)
    sphere_volume = calculate_sphere_volume(number_of_protons, number_of_neutrons)
    return sphere_volume / initial_volume


def calculate_radius(number_of_protons: int, number_of_neutrons: int, beta_values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calculate nuclear radius as a function of polar angle theta."""
    radius = np.ones_like(theta)
    number_of_nucleons: int = number_of_protons * number_of_neutrons

    for harmonic_index in range(1, 13):
        harmonic = np.real(sph_harm_y(harmonic_index, 0, theta, 0.0))
        radius += beta_values[harmonic_index - 1] * harmonic

    volume_fix = calculate_volume_fixing_factor(number_of_protons, number_of_neutrons, beta_values) ** (1 / 3)
    return r0 * (number_of_nucleons ** (1 / 3)) * volume_fix * radius


def check_neck(number_of_protons: int, number_of_neutrons: int, beta_values: np.ndarray, neck_condition: float = 2.0) -> bool:
    """Check if nucleus has a neck smaller than a given condition."""
    theta: np.ndarray = np.linspace(0, np.pi, 1800)
    sin_radius = calculate_radius(number_of_protons, number_of_neutrons, beta_values, theta) * np.sin(theta)

    if np.min(sin_radius) < neck_condition:
        return True
    else:
        return False
