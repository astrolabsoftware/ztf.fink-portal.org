import io
import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from scipy.spatial.transform import Rotation
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
from fink_utils.sso.spins import cos_aspect_angle, rotation_phase, subobserver_longitude
from fink_utils.sso.utils import compute_light_travel_correction
from fink_utils.sso.miriade import query_miriade
from functools import lru_cache
from app import app

TWO_PI = 2 * np.pi
deg = np.degrees


def asteroid_3d(): # cf. Sec. 2.1
    return html.Div([
        dcc.Store(id="model-3d-params"),
        dcc.Store(id="model-3d-jd"),

        # Message d'information
        html.Div([
            DashIconify(icon="tabler:alert-circle", width=20, color="#d9534f"),
            html.Span("The latest observation is selected by default. "
                      "Click on a point in the lightcurve to select another date."),
        ], className="model3d-instruction"),

        # Modele 3D + elements superposes
        html.Div([
            html.Div(id="model-3d", style={"width": "100%", "height": "450px"}),
            html.Div([
                dcc.Dropdown(id="model-3d-date-format", value="iso", clearable=False, searchable=False,
                             options=[{"label": "ISO", "value": "iso"}, {"label": "JD", "value": "jd"}, {"label": "MJD", "value": "mjd"}],
                             className="model3d-date-select"),
                html.Span(id="model-3d-date", className="model3d-date-value"),
            ], className="model3d-date-bar"),
            html.Button(DashIconify(icon="majesticons:globe-grid-line", width=23), id="model3d-coordinates-button",
                        className="model3d-coordinates-button", title="Show/hide latitude and longitude grid"),
            html.Button(DashIconify(icon="mdi:axis-arrow", width=23), id="model3d-axes-button",
                         className="model3d-axes-button", title="Show/hide asteroid axes"),
            html.Button(DashIconify(icon="tabler:hand-move", width=21), id="model3d-rotate-button",
                        className="model3d-rotate-button", title="Enable manual rotation"),
            html.Button(DashIconify(icon="mdi:play", width=23), id="model3d-play-button",
                        className="model3d-play-button", title="Play / Pause asteroid rotation"),
            # Boutons temporaires de comparaison avec DAMIT (absents de la version finale)
            html.Button("P", id="model3d-periode-button", className="model3d-periode-button",
            title="Period correction"),
            html.Button("D", id="model3d-damit-button", className="model3d-damit-button",
                        title="DAMIT correction"),
            html.Button(DashIconify(icon="tabler:arrows-maximize", width=19), id="model3d-fullscreen",
                        className="model3d-fullscreen", title="Fullscreen"),
            html.Div([html.Button("+", id="model3d-zoom-in"), html.Button("−", id="model3d-zoom-out")],
                     className="model3d-zoom-controls"),
            html.Div("Camera FoV: 25°", className="model3d-fov"),
        ], className="model3d-view"),

        # Liens OBJ
        html.Div([
            html.A("View OBJ", id="model3d-view-obj", href="#", target="_blank", className="model3d-download-link"),
            html.A("Download OBJ", id="model3d-download-obj", href="#", download="asteroid.obj", className="model3d-download-link"),
        ], className="model3d-download"),

    ], className="model3d-container")


def unit(v):
    return v / np.linalg.norm(v)

def get_ephemeris_position(ssoid, date):  # cf. Sec. 2.2
    jd = Time(date, format="isot", scale="utc").jd
    eph = query_miriade(ident=str(ssoid), jd=[jd], observer="500", shift=0)  # observer 500 = géocentre
    if eph.empty:
        raise ValueError(f"No Miriade ephemerides for {ssoid} at {date}")
    # position de l'astéroïde (RA, DEC), du Soleil vu de l'astéroïde (RA_h, DEC_h) et distance à l'observateur
    ra, dec, ra_sun, dec_sun, d_obs = (float(eph.iloc[0][k]) for k in ("RA", "DEC", "RA_h", "DEC_h", "Dobs"))
    return ra, dec, jd, ra_sun, dec_sun, d_obs

def compute_orientation(alpha0, delta0, W0, W1, t0, ra, dec, jd, d_obs): # cf. Sec. 2.4
    # retourne (angle d'aspect, position angle du pole, longitude sub-observateur), en radians
    a0, d0, ra, dec = np.radians([alpha0, delta0, ra, dec])
    ra, dec = (ra + np.pi) % TWO_PI, -dec  # on inverse le sens de visee (asteroide -> Terre), Eq. (7)

    aspect = np.arccos(np.clip(cos_aspect_angle(ra, dec, a0, d0), -1, 1))
    W = rotation_phase(compute_light_travel_correction(jd, d_obs), W0, W1, t0)
    phi = (np.pi - subobserver_longitude(ra, dec, a0, d0, W)) % TWO_PI #Eq. (8), conversion de convention fink -> 3D

    y = -np.cos(d0) * np.sin(ra - a0)   # Eq. (10)
    x = np.sin(d0) * np.cos(dec) - np.cos(d0) * np.sin(dec) * np.cos(ra - a0) # Eq. (9)
    return aspect, np.arctan2(y, x), phi # Eq. (11)


def orientation_to_quaternion(aspect, np_angle, phi):  # cf. Sec. 2.5
    # Repere camera Three.js : +X droite, +Y haut, +Z vers l'observateur
    z = unit([np.sin(aspect) * np.sin(np_angle), np.sin(aspect) * np.cos(np_angle), np.cos(aspect)]) # Eq. (12), où doit pointer z_ast dans la caméra
    e0 = unit(np.array([0.0, 0.0, 1.0]) - z[2] * z)  # Eq. (13), direction auxiliaire
    e1 = unit(np.cross(z, e0)) # Eq. (14), direction auxiliaire
    x = np.cos(phi) * e0 - np.sin(phi) * e1 # Eq. (15),  où doit pointer x_ast dans la caméra
    y = np.cross(z, x) # Eq. (16),  où doit pointer y_ast dans la caméra
    return Rotation.from_matrix(np.column_stack([x, y, z])).as_quat().tolist()  # matrice de rotation repere asteroide -> repere camera, puis conversion en quaternion


def radec_to_vector(ra, dec):  # cf. Sec. 2.6
    ra, dec = np.radians([ra, dec])
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]) # Eq. (17)


def sun_to_camera_vector(ra, dec, ra_sun, dec_sun):  # cf. Sec. 2.7
    # base d'observation construite dans le repere du ciel
    z = radec_to_vector(ra, dec) # Eq. (18), axe Z suivant la ligne de visee
    x = unit(np.cross([0.0, 0.0, 1.0], z))   # Eq. (19), axe X dans le plan du ciel
    y = unit(np.cross(z, x))   # Eq. (20), axe Y dans le plan du ciel
    return unit(np.array([x, y, z]) @ radec_to_vector(ra_sun, dec_sun)) # Eq. (21), projection de la direction de l'asteroide vu depuis le soleil dans le repere de la camera

@lru_cache(maxsize=128)
def get_damit_major_axis_angle(ssoid):  # cf. Sec. 2.8
    # Recherche du modele DAMIT correspondant a l'asteroide (None si absent ou service indisponible)
    try:
        r = requests.get("https://ssp.imcce.fr/webservices/miriade/api/ephemph.php",
                         params={"-get": "model", "-name": f"a:{ssoid}", "-mime": "json"}, timeout=20)
        if r.status_code == 404:  # Miriade ne connait aucun modele pour cet objet
            return None
        r.raise_for_status()
        models = r.json()
        model = next((m for m in models if m.get("ssoModel") == "DAMIT" and m.get("ssoShapeVerFile")), None)
        if model is None:
            return None

        # Telechargement du shape.obj et extraction des sommets (x, y)
        url = ("https://astro.troja.mff.cuni.cz/projects/damit/generated_files/open/AsteroidModel/"
               f"{model['ssoModelDbId']}/shape.obj")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        xy = np.array([l.split()[1:3] for l in r.text.splitlines() if l.startswith("v ")], dtype=float)
        if len(xy) < 3:
            return None

        # Angle du grand axe dans le plan XY + parametres de spin DAMIT
        _, vecs = np.linalg.eigh(np.cov(xy.T))
        return {
            "offset": deg(np.arctan2(vecs[1, -1], vecs[0, -1])) % 180,
            "alpha0": model["iauSpin_RA (deg)"],
            "delta0": model["iauSpin_DEC (deg)"],
            "W0": np.radians(model["iauSpin_W (deg)"]),
            "W1": np.radians(model["iauSpin_Wp (deg/d)"]),
            "t0": model["iauSpin_TRef"],
        }
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as e:
        print(f"DAMIT indisponible pour {ssoid} : {e}")
        return None

def build_3d_orientation(ssoid, socca, date):  # cf. Sec. 2.9
    ra, dec, jd, ra_sun, dec_sun, d_obs = get_ephemeris_position(ssoid, date)
    t0, P_socca = 2459580.5, socca["period"]

    # Orientation SOCCA
    aspect, np_angle, phi_socca = compute_orientation(
        socca["alpha0"], socca["delta0"], socca["phi0"],
        TWO_PI / P_socca, t0, ra, dec, jd, d_obs,
    )
    phi_corrected = phi_socca
    phi_periode = phi_socca 

    d = get_damit_major_axis_angle(ssoid)
    if d is None:
        print(f"Aucun modele DAMIT pour {ssoid} : orientation SOCCA uniquement.")
    else:
        P_damit = TWO_PI / d["W1"]
        print(f"DAMIT ({ssoid}): alpha0={d['alpha0']:.4f} delta0={d['delta0']:.4f} "
              f"W0={deg(d['W0']):.4f} W1={deg(d['W1']):.6f}°/j t0={d['t0']} "
              f"P={P_damit * 24:.6f} h offset={d['offset']:.4f}")

        # Orientation DAMIT
        asp_d, _, phi_damit = compute_orientation(
            d["alpha0"], d["delta0"], d["W0"], d["W1"], d["t0"], ra, dec, jd, d_obs,
        )

        offset = np.radians(d["offset"])                      # décalage de repère
        delta_period = TWO_PI * (compute_light_travel_correction(jd, d_obs) - t0) * (1 / P_damit - 1 / P_socca)          # correction de période
        phi_socca_in_damit = (phi_socca + offset) % TWO_PI
        phi_corrected = (phi_socca_in_damit - delta_period) % TWO_PI
        phi_periode = (phi_socca - delta_period) % TWO_PI
        print(f"DAMIT : aspect = {deg(asp_d):.2f}, lon = {deg(phi_damit):.2f}, lat = {90 - deg(asp_d):.2f}")
        print(f"SOCCA : aspect = {deg(aspect):.2f}, lon = {deg(phi_socca):.2f}, lat = {90 - deg(aspect):.2f}")
        print(f"Période : correction = {deg(delta_period) % 360:.2f}, "
            f"phi SOCCA (repère DAMIT) {deg(phi_socca_in_damit):.2f} -> {deg(phi_corrected):.2f} "
            f"(phi DAMIT = {deg(phi_damit):.2f})")

    q_socca = orientation_to_quaternion(aspect, np_angle, phi_socca)
    q_periode = orientation_to_quaternion(aspect, np_angle, phi_periode) if d is not None else None 
    q_damit = orientation_to_quaternion(aspect, np_angle, phi_corrected) if d is not None else None
    return q_socca, q_periode, q_damit, sun_to_camera_vector(ra, dec, ra_sun, dec_sun).tolist()

@app.callback(
    Output("model-3d-params", "data"),
    Output("model-3d-jd", "data"),
    Input("socca-params", "data"),
    Input("sso-lightcurve", "clickData"),
    State("object-sso", "data"),
    prevent_initial_call=True,
)
def update_3d(socca, click_data, object_sso):
    if not socca or not object_sso:
        raise PreventUpdate

    pdf = pd.read_json(io.StringIO(object_sso))
    ssoid = str(pdf["i:ssnamenr"].iloc[0])  # Recuperation de l'identifiant de l'asteroide

    jd = float(pdf["i:jd"].max())  # Derniere observation par defaut
    if click_data and click_data.get("points"):
        jd = float(click_data["points"][0]["customdata"]) + 2400000.5  # point clique (MJD -> JD)

    q_socca, q_periode, q_damit, sun_direction = build_3d_orientation(ssoid, socca, Time(jd, format="jd", scale="utc").isot)
    return {"quaternion": q_socca, "quaternion_periode": q_periode, "quaternion_damit": q_damit, "sun_direction": sun_direction,
            "a_b": socca["a_b"], "a_c": socca["a_c"], "period": socca["period"]}, jd

@app.callback(
    Output("model-3d-date", "children"),
    Input("model-3d-jd", "data"),
    Input("model-3d-date-format", "value"),
)
def display_3d_date(jd, date_format):
    if jd is None:
        return ""
    if date_format == "jd":
        return f"{jd:.7f}"
    if date_format == "mjd":
        return f"{jd - 2400000.5:.7f}"
    return Time(jd, format="jd", scale="utc").isot.replace("T", " ") + " UTC"


app.clientside_callback(
    """
    function(p) {
        if (!p) return "";
        window.dash_clientside.model3d.draw(p);
        return "";
    }
    """,
    Output("model-3d", "title"),
    Input("model-3d-params", "data"),
) # appel du rendu Three.js lorsque les parametres sont mis a jour