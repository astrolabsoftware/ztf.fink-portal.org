# Copyright 2025 AstroLab Software
# Authors: Julian Hamo, Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from astropy.coordinates import SkyCoord, EarthLocation, get_body, Latitude, Longitude
import numpy as np
import astroplan as apl
import astropy.units as u
from astropy.time import Time
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import datetime

from fink_utils.sso.miriade import query_miriade

night_colors = [
    "rgba(204, 229, 255, 0.5)",
    "rgba(153, 204, 255, 0.5)",
    "rgba(77, 136, 255, 0.5)",
    "rgba(26, 26, 255, 0.5)",
    "rgba(77, 136, 255, 0.5)",
    "rgba(153, 204, 255, 0.5)",
    "rgba(204, 229, 255, 0.5)",
]

moon_color = "#9900cc"

additional_observatories_data = {
    "Caucasian Mountain Observatory": {
        "lon_deg": "43:44:10",
        "lat_deg": "+42:40:03",
        "height_meter": 2112,
    }
}

additional_observatories = {
    name: EarthLocation.from_geodetic(
        lat=Latitude(additional_observatories_data[name]["lat_deg"], unit=u.deg).deg,
        lon=Longitude(additional_observatories_data[name]["lon_deg"], unit=u.deg).deg,
        height=additional_observatories_data[name]["height_meter"] * u.m,
    )
    for name in additional_observatories_data.keys()
}


def observation_time_to_utc_offset(observatory):
    """Compute the timezone offset from the observatory location to UTC.

    Parameters
    ----------
    observatory: astropy.coordinates.EarthLocation
        Astropy EarthLocation object representing the observatory

    Returns
    -------
    offset: float
        Time difference between observatory local time zone and UTC (in hour).
    """
    lat, lon = observatory.lat.deg, observatory.lon.deg
    tz = TimezoneFinder().timezone_at(lat=lat, lng=lon)
    offset = (
        datetime.datetime.now()
        .replace(tzinfo=ZoneInfo("UTC"))
        .astimezone(ZoneInfo(tz))
        .utcoffset()
        .total_seconds()
        // 3600
    )

    return offset


def observation_time(date, delta_points=0.25):
    """Return an astropy.time.Time array of time starting from -12h to +12h around the date. Points are separated by delta_time.

    Parameters
    ----------
    date: str
        Considered date for observation. Format in YYYY-MM-DD.
    delta_points: float, optional (default=0.25)
        Time elapsed between two points

    Returns
    -------
    obs_time: np.array[astropy.time.Time]
        Array of time points starting from -12h to +12h.
    """
    obs_time = (
        Time(date, scale="utc")
        + np.linspace(-12, 12 - delta_points, int(24 / delta_points)) * u.hour
    )

    return obs_time


def target_coordinates(ra, dec, observatory, obs_time):
    """Compute the coordinates of a source from an observatory at each time of obs_time.

    Parameters
    ----------
    ra: float
        Right ascension of the source, in degree
    dec: float
        Declination of the source, in degree
    observatory: astropy.coordinates.EarthLocation
        Astropy EarthLocation object representing the observatory
    obs_time: np.array[astropy.time.Time]
        Array of times for each point

    Returns
    -------
    coordinates: np.array[astropy.coordinates.SkyCoord]
        Coordinates of the source (with elevation and azimut)
        from the observatory at every point in obs_time
    """
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    observer = apl.Observer(location=observatory)
    coordinates = observer.altaz(obs_time, target=target)

    return coordinates


def moon_coordinates(observatory, obs_time):
    """Compute the coordinates of the Moon from an observatory at each time of obs_time.

    Parameters
    ----------
    observatory: astropy.coordinates.EarthLocation
        Astropy EarthLocation object representing the observatory
    obs_time: np.array[astropy.time.Time]
        Array of times for each point

    Returns
    -------
    coordinates: np.array[astropy.coordinates.SkyCoord]
        Coordinates of the Moon (with elevation and azimut) from
        the observatory at every point in obs_time
    """
    observer = apl.Observer(location=observatory)
    coordinates = observer.moon_altaz(obs_time)

    return coordinates


def from_elevation_to_airmass(elevation):
    """Compute the relative airmass from the elevation

    Notes
    -----
    1 relative airmass is the airmass at a 90 degree angle of elevation

    Parameters
    ----------
    elevation: np.array[astropy.units.Angle]
        Array of elevations in degrees

    Returns
    -------
    out: np.array
        Array of relative airmass
    """
    return 1 / np.cos(np.radians(90 - elevation))


def get_moon_phase(time):
    """Retrieve the unicode symbol for the Moon phase.

    Parameters
    ----------
    time: str
        Considered date for observation. Format in YYYY-MM-DD.

    Returns
    -------
    out: str
        Unicode symbol representing the phase of the Moon at the considered date.
    """
    # Moon angle of illumination
    phase_angle = apl.moon_phase_angle(time).value * 180 / np.pi
    elongation = get_body("moon", time).ra - get_body("sun", time).ra
    if elongation < 0:
        elongation += 360 * u.deg

    # New Moon
    if phase_angle > 170:
        return "\U0001f311"

    # Full Moon
    elif phase_angle < 10:
        return "\U0001f315"

    else:
        # Waxing Moon
        if elongation.value < 180:
            # Waxing Crescent
            if phase_angle >= 100:
                return "\U0001f312"

            # First Quarter
            elif phase_angle > 80:
                return "\U0001f313"

            # Waxing Gibbous
            else:
                return "\U0001f314"

        # Waning Moon
        else:
            # Waning Gibbous
            if phase_angle <= 80:
                return "\U0001f316"

            # Last Quarter
            elif phase_angle < 100:
                return "\U0001f317"

            # Waning Crescent
            else:
                return "\U0001f318"


def get_moon_illumination(time):
    """Compute Moon illumination at the considered time.

    Parameters
    ----------
    time: str
        Considered date for observation. Format in YYYY-MM-DD.

    Returns
    -------
    out: float
        Moon illumination fraction (0 is new Moon and 1 is full Moon)
        at the considered date.
    """
    return apl.moon_illumination(time)


def utc_night_hours(observatory, date, offset, UTC=False):
    """Time of the different definitions of twilight and dawn from an observatory at a given date.

    Parameters
    ----------
    observatory: astropy.coordinates.EarthLocation
        Astropy EarthLocation object representing the observatory
    date: str
        Considered date for observation. Format in YYYY-MM-DD.
    offset: float
        Time difference between observatory local time zone and UTC (in hour).
    UTC: bool (optional, default=False)
        If UTC, the twilights are computed in the UTC scale.

    Returns
    -------
    twilights: dict[astropy.time.Time]
        Dictionary with:
        - Sunset time
        - Civil twilight
        - Nautical twilight
        - Astronomical twilight
        - Astronomical morning
        - Nautical morning
        - Civil morning
        - Sunrise
    """
    offset_UTC = offset
    if UTC:
        offset_UTC = 0
    observer = apl.Observer(location=observatory)

    twilights = {
        "Sunset": observer.sun_set_time(Time(date) - offset * u.hour, which="previous")
        + offset_UTC * u.hour,
        "Civil twilight": observer.twilight_evening_civil(
            Time(date) - offset * u.hour, which="previous"
        )
        + offset_UTC * u.hour,
        "Nautical twilight": observer.twilight_evening_nautical(
            Time(date) - offset * u.hour, which="previous"
        )
        + offset_UTC * u.hour,
        "Astronomical twilight": observer.twilight_evening_astronomical(
            Time(date) - offset * u.hour, which="previous"
        )
        + offset_UTC * u.hour,
        "Astronomical morning": observer.twilight_morning_astronomical(
            Time(date) - offset * u.hour, which="next"
        )
        + offset_UTC * u.hour,
        "Nautical morning": observer.twilight_morning_nautical(
            Time(date) - offset * u.hour, which="next"
        )
        + offset_UTC * u.hour,
        "Civil morning": observer.twilight_morning_civil(
            Time(date) - offset * u.hour, which="next"
        )
        + offset_UTC * u.hour,
        "Sunrise": observer.sun_rise_time(Time(date) - offset * u.hour, which="next")
        + offset_UTC * u.hour,
    }
    return twilights


def from_time_to_axis(times):
    """Tranform an astropy.time.Time array into an array of hours starting at -12h.

    Parameters
    ----------
    times: np.array[astropy.time.Time]
        Array of time for observation

    Returns
    -------
    axis: np.array
        List of hours starting at -12h
    """
    return np.array([time.to_value("iso", subfmt="date_hm")[-5:] for time in times])


# Copied from apps.summary: think of a better storage solution
def is_sso(pdfs):
    """Auxiliary function to check whether the object is a SSO"""
    payload = pdfs["i:ssnamenr"].to_numpy()[0]
    if str(payload) == "null" or str(payload) == "None":
        return False

    if np.all([i == payload for i in pdfs["i:ssnamenr"].to_numpy()]):
        return True

    return False
    

def sso_coordinates(pdf, time):
    ssnamenr = pdf["i:ssnamenr"].unique().astype(str)
    if len(ssnamenr) > 1:
        print(
            f"""
Error: the object is associated to multiple known SSOs \
- Selecting only the first identifier {ssnamenr[0]}.
            """
        )
        ssnamenr = ssnamenr[0]
    else:
        ssnamenr = ssnamenr[0]
    miriade_data = query_miriade(
        ssnamenr, time,
        observer="I41",
        rplane="1",
        tcoor=5,
        shift=15.0,
        timeout=30,
        return_json=True,
        iofile="ephemcc-photom.xml",
    )
    ra = np.full(len(time), np.nan)
    dec = np.full(len(time), np.nan)
    for idx, d in enumerate(miriade_data["data"]):
        ra[idx] = d["RA"]
        dec[idx] = d["DEC"]
    return ra, dec
