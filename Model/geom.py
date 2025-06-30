"""
Geometry helpers – great-circle distance & cumulative travel-time.
"""
from math import radians, sin, cos, atan2, sqrt

R_EARTH = 6371.0    # km

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in *kilometres*."""
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1;  dλ = λ2 - λ1
    a  = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2 * R_EARTH * atan2(sqrt(a), sqrt(1-a))

def travel_time_km(distance_km: float,
                   top_kph: float,
                   accel: float,
                   decel: float) -> float:
    """
    Very simple trapezoidal speed profile: accel → cruise → decel.
    Returns *minutes*.
    """
    vmax_mps = top_kph * 1000/3600
    t_acc    = vmax_mps / accel
    s_acc_km = 0.5 * accel * t_acc**2 / 1000
    t_dec    = vmax_mps / decel
    s_dec_km = 0.5 * decel * t_dec**2 / 1000

    if distance_km < s_acc_km + s_dec_km:      # never hits vmax
        vmax_mps = (accel * decel * distance_km * 1000 /
                   (accel + decel))**0.5
        t = vmax_mps/accel + vmax_mps/decel
    else:
        s_cruise = distance_km - s_acc_km - s_dec_km
        t = t_acc + t_dec + (s_cruise / (top_kph/60))   # cruise segment
    return t
