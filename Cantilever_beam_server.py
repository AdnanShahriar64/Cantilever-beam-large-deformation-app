# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 23:49:02 2026

@author: AluPotol
"""

import numpy as np

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go


# ==========================================================
# YOUR SOLVER FUNCTIONS (keep logic)
# ==========================================================

def inertia_and_distributed_load_generator(Domain, CS_Geometry, Case, rho, Distributed_Load, ds, Gravity):
    Beam_Length = CS_Geometry[0]
    Height_at_tip = CS_Geometry[1]
    Height_at_fixed_end = CS_Geometry[2]
    Width_at_tip = CS_Geometry[3]
    Width_at_fixed_end = CS_Geometry[4]
    Hollow_section_height_at_tip = CS_Geometry[5]
    Hollow_section_height_at_fixed_end = CS_Geometry[6]
    Hollow_section_width_at_tip = CS_Geometry[7]
    Hollow_section_width_at_fixed_end = CS_Geometry[8]
    Outer_diameter_at_tip = CS_Geometry[9]
    Hollow_section_diameter_at_tip = CS_Geometry[10]
    Outer_diameter_at_fixed_end = CS_Geometry[11]
    Hollow_section_diameter_at_fixed_end = CS_Geometry[12]

    F = np.zeros(len(Domain))
    g = Gravity

    if Case == 0:
        H = Domain * (Height_at_fixed_end - Height_at_tip) / Beam_Length + Height_at_tip
        W = Domain * (Width_at_fixed_end - Width_at_tip) / Beam_Length + Width_at_tip
        h = Domain * (Hollow_section_height_at_fixed_end - Hollow_section_height_at_tip) / Beam_Length + Hollow_section_height_at_tip
        w = Domain * (Hollow_section_width_at_fixed_end - Hollow_section_width_at_tip) / Beam_Length + Hollow_section_width_at_tip

        I = (W * H**3 - w * h**3) / 12.0
        F[:] = ((H - h) * (W - w) * ds) * rho * g + Distributed_Load * ds

        # [-outer/2, -inner/2, +inner/2, +outer/2]
        Y_Range = np.column_stack((-H/2, -h/2, h/2, H/2)).astype(float)

    elif Case == 1:
        D = Domain * (Outer_diameter_at_fixed_end - Outer_diameter_at_tip) / Beam_Length + Outer_diameter_at_tip
        d = Domain * (Hollow_section_diameter_at_fixed_end - Hollow_section_diameter_at_tip) / Beam_Length + Hollow_section_diameter_at_tip

        I = (D**4 - d**4) * np.pi / 64.0
        F[:] = ((np.pi * (D**2 - d**2)) * ds) * rho * g + Distributed_Load * ds

        Y_Range = np.column_stack((-D/2, -d/2, d/2, D/2)).astype(float)

    else:
        raise NotImplementedError("Only Case 0 (rect) and Case 1 (circular) implemented.")

    # smoothing
    I_sm = I.copy()
    I_sm[1:-1] = 0.5 * (I_sm[:-2] + I_sm[2:])
    return I_sm, F, Y_Range


def theta_calculator(ThetaInt, ds, E, I, Mext, F, P, Domain):
    N = len(Domain)
    M = np.zeros(N)
    Theta = np.zeros(N)

    Theta[0] = ThetaInt
    M = M + Mext

    CumF = np.cumsum(F)
    CumP = np.cumsum(P)

    for i in range(1, N):
        ct = np.cos(Theta[i - 1])
        M[i] = (
            M[i - 1]
            + CumF[i - 1] * ds * ct
            + (F[i] * ds * ct) / 2.0
            + CumP[i - 1] * ds * ct
        )
        Theta[i] = Theta[i - 1] + M[i] * (ds / (E * I[i - 1]))

    return Theta, M


def solve_beam(
    Beam_Length,
    Case,
    CS_Geometry,
    E,
    rho,
    Concentrated_loadings,
    Concentrated_moments,
    Distributed_Load,
    Gravity,
    Number_of_divisions=1000
):
    
    status_log = []
    
    Domain = np.linspace(0.0, Beam_Length, Number_of_divisions + 1)
    ds = Domain[1] - Domain[0]
    N = len(Domain)

    
    def nearest_index(domain: np.ndarray, x: float) -> int:
        return int(np.argmin(np.abs(domain - x)))

    # concentrated forces
    P = np.zeros(N)
    for x_pos, load_val in Concentrated_loadings:
        idx = nearest_index(Domain, x_pos)
        P[idx] += load_val

    # concentrated moments
    Mext = np.zeros(N)
    for x_pos, m_val in Concentrated_moments:
        idx = nearest_index(Domain, x_pos)
        Mext[idx] += m_val
        
    msg ="Set concentrated loads and moments"
    status_log.append(msg)

    I, F, Y_Range = inertia_and_distributed_load_generator(
        Domain, CS_Geometry, Case, rho, Distributed_Load, ds, Gravity
    )

    msg ="Obtained moment of inertia and distributed loads"
    status_log.append(msg)

    msg ="------- Solver ----------"
    status_log.append(msg)
    # bisection
    Tol = 1e-10
    ThetaPos = np.pi / 2.0
    ThetaNeg = -np.pi / 2.0

    


    for C in range(1, 501):
        ThetaMid = 0.5 * (ThetaPos + ThetaNeg)
        Theta, M = theta_calculator(ThetaMid, ds, E, I, Mext, F, P, Domain)

        if Theta[-1] > 0:
            ThetaPos = ThetaMid
        elif Theta[-1] < 0:
            ThetaNeg = ThetaMid
            
        if abs(Theta[-1]) < Tol:   
            msg = f"Iteration={C}   Theta(end)={Theta[-1]:.12e}"
            status_log.append(msg)            
            break
         

        msg = f"Iteration={C}   Theta(end)={Theta[-1]:.12e}"
        status_log.append(msg)
        

        
        
    msg = "-------------------------"
    status_log.append(msg)        
    
    if abs(Theta[-1]) > Tol:
        msg = f"Warning! Couldn't converge. Boundary Angle = {Theta[-1]:.12e}"
    else:
        msg =f"Solver finished with Boundary Angle={Theta[-1]:.12e}"
        
    status_log.append(msg) 
           
    

    return Domain, ds, Theta, M, I, Y_Range, status_log



def postprocess(Domain, ds, Theta, M, I, Y_Range, DP=51):
    N = len(Domain)

    # centerline
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(1, N):
        x[i] = x[i-1] + ds * np.cos(Theta[i-1])
        y[i] = y[i-1] + ds * np.sin(Theta[i-1])

    # rescale
    x = (-x[-1] + x)
    y = (y - y[-1])

    # fields on deformed coordinates
    Stress = np.zeros((DP, N))
    Ph = np.zeros((DP, N))
    Px = np.zeros((DP, N))

    for i in range(1, N):
        YR = np.linspace(Y_Range[i, 0], Y_Range[i, -1], DP)
        Stress[:, i] = -(M[i] / I[i-1]) * YR

        # If you intentionally "blank out" a hollow band in y, do it here.
        # NOTE: This is NOT physically correct for circular tubes (radial void),
        # but you said you want it case-independent for plotting.
        hole = (YR > Y_Range[i, 1]) & (YR < Y_Range[i, 2])
        Stress[hole, i] = 0.0

        Ph[:, i] = YR + y[i] + YR * Theta[i] * np.sin(Theta[i])
        Px[:, i] = x[i] - YR * Theta[i] * np.cos(Theta[i])

    # i=0 section coords (stress left at 0 like MATLAB)
    YR0 = np.linspace(Y_Range[0, 0], Y_Range[0, -1], DP)
    Ph[:, 0] = YR0 + y[0] + YR0 * Theta[0] * np.sin(Theta[0])
    Px[:, 0] = x[0] - YR0 * Theta[0] * np.cos(Theta[0])

    return x, y, Px, Ph, Stress


# ==========================================================
# DASH APP
# ==========================================================

app = Dash(__name__)

server = app.server


app.title = "Large-Deformation Beam (2D)"

def default_loads_text():
    return "0.0, 1000\n5.0, 0.0"

def default_moments_text():
    return "0.0, 0.0\n7.5, 0.0"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "20px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Large-Deformation Cantilever Beam Simulator (2D)"),

        html.H5("Reference paper: https://www.lhscientificpublishing.com/Journals/articles/DOI-10.5890-DNC.2024.06.012.aspx"),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "14px"},
            children=[
                html.Div([
                    
                    html.H4("Set Material properties"),
                    html.Label("Modulus of elasticity"),
                    dcc.Input(id="E", type="number", value=30e9, step=1e9, style={"width":"100%"}),

                    html.Br(),
                    html.Label("Density"),
                    dcc.Input(id="rho", type="number", value=2700.0, step=10, style={"width":"100%"}),

                    html.H4("Loads (Beam tip position: 0)"),
                    html.Label("Distributed Load (N/m)"),
                    dcc.Input(id="Distributed_Load", type="number", value=1000.0, step=10, style={"width":"100%"}),

                    html.Br(),
                    html.Label("Gravity (m/s^2)"),
                    dcc.Input(id="Gravity", type="number", value=-9.81, step=0.01, style={"width":"100%"}),
                    
                    html.Br(),
                    html.Label("Concentrated Loads (Load position, Value)"),
                    dcc.Textarea(id="loads_text", value=default_loads_text(), style={"width":"100%","height":"90px"}),

                    html.Br(),
                    html.Label("Concentrated Moments (Load position, Value)"),
                    dcc.Textarea(id="moments_text", value=default_moments_text(), style={"width":"100%","height":"90px"}),                    
                    
                ]),

                html.Div([
                    
                    html.H4("Set Beam Geometry"),
                    html.Label("Cross section type"),
                    dcc.Dropdown(
                        id="Case",
                        options=[
                            {"label": "Rectangular", "value": 0},
                            {"label": "Circular", "value": 1},
                        ],
                        value=1,
                        clearable=False
                    ),                  
                    
                    html.Br(), html.Br(),
                    html.Label("Beam Length"),
                    dcc.Input(id="Beam_Length", type="number", value=10.0, step=0.1, style={"width":"100%"}),

                    
                    html.H4("Circular Geometry"),                  
                    html.Label("D (CS outer diameter)"),
                    
                    
                    # dcc.Input(id="D_tip", type="number", value=0.05, step=0.005, style={"width":"60%"}),
                    html.Br(),html.Br(),
                    html.Div([
                        html.Div("At tip", style={"width": "40%"}),
                        dcc.Input(id="D_tip", type="number", value=0.05, step=0.005, style={"width": "60%"})
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                    
                    html.Div([
                        html.Div("At fixed end", style={"width": "40%"}),
                        dcc.Input(id="D_fix", type="number", value=0.15, step=0.005, style={"width": "60%"})
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
                   
                    
                    # dcc.Input(id="D_fix", type="number", value=0.15, step=0.005, style={"width":"60%"}),

                    html.Br(), html.Br(),
                    html.Label("d (Hollow section diameter)"),
                    html.Br(),html.Br(),
                    html.Div([
                        html.Div("At tip", style={"width": "40%"}),
                        dcc.Input(id="d_tip", type="number", value=0.01, step=0.005, style={"width": "60%"})
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                    
                    html.Div([
                        html.Div("At fixed end", style={"width": "40%"}),
                        dcc.Input(id="d_fix", type="number", value=0.05, step=0.005, style={"width": "60%"})
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

                    
                    # html.Br(),
                    # html.Div(id="status", style={"marginTop":"10px", "color":"#444"}),
                ]),

                html.Div([

                    
                html.H4("Rectangular Geometry"),
                
                html.Label("H (CS height)"),
                html.Br(),html.Br(),
                html.Div([
                    html.Div("At tip", style={"width": "40%"}),
                    dcc.Input(id="H_tip", type="number", value=0.05, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                
                html.Div([
                    html.Div("At fixed end", style={"width": "40%"}),
                    dcc.Input(id="H_fix", type="number", value=0.15, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                
                
                html.Label("W (CS width)"),
                html.Br(),html.Br(),
                html.Div([
                    html.Div("At tip", style={"width": "40%"}),
                    dcc.Input(id="W_tip", type="number", value=0.05, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                
                html.Div([
                    html.Div("At fixed end", style={"width": "40%"}),
                    dcc.Input(id="W_fix", type="number", value=0.15, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                
                
                html.Label("h (Hollow section height)"),
                html.Br(),html.Br(),
                html.Div([
                    html.Div("At tip", style={"width": "40%"}),
                    dcc.Input(id="h_tip", type="number", value=0.01, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                
                html.Div([
                    html.Div("At fixed end", style={"width": "40%"}),
                    dcc.Input(id="h_fix", type="number", value=0.05, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                
                
                html.Label("w (Hollow section width)"),
                html.Br(),html.Br(),
                html.Div([
                    html.Div("At tip", style={"width": "40%"}),
                    dcc.Input(id="w_tip", type="number", value=0.01, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                
                html.Div([
                    html.Div("At fixed end", style={"width": "40%"}),
                    dcc.Input(id="w_fix", type="number", value=0.05, step=0.005, style={"width": "60%"})
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),


                ]),
            ]
        ),


        
        # html.Button("Solve", id="run_btn", n_clicks=0, style={"padding":"20px 150px"}),                    



        html.Div([
            
            # Solve button (spans 1 column)
            html.Div(
                html.Button(
                    "Solve",
                    id="run_btn",
                    n_clicks=0,
                    style={
                        "padding": "20px 60px",
                        "width": "100%"
                    }
                ),
                style={"gridColumn": "span 1"}
            ),
        
            # Status box (spans 2 columns)
            html.Div(
                html.Div(
                    id="status_box",
                    style={
                        "whiteSpace": "pre-wrap",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "height": "80px",
                        "overflowY": "scroll",
                        "fontFamily": "monospace",
                        "backgroundColor": "#f8f8f8"
                    }
                ),
                style={"gridColumn": "span 2"}
            )
        
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr 1fr",
            "gap": "10px",
            "marginTop": "15px"
        }),



        dcc.Graph(id="stress_plot", style={"height":"650px"}),

        html.Div(id="summary", style={"marginTop":"10px"})
    ]
)


def parse_csv_pairs(text):
    """
    Expect lines: x, value
    Empty lines ignored.
    """
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad line: '{line}' (expected: x, value)")
        rows.append([float(parts[0]), float(parts[1])])
    if not rows:
        return np.zeros((0,2), dtype=float)
    return np.array(rows, dtype=float)


def curved_stress_scatter(fig, Px, Ph, Stress, downsample=1):
    X = Px[:, ::downsample].ravel()
    Y = Ph[:, ::downsample].ravel()
    S = Stress[:, ::downsample].ravel()

    fig.add_trace(go.Scattergl(
        x=X,
        y=Y,
        mode="markers",
        marker=dict(
            size=3,
            color=S,
            colorscale="Jet",
            colorbar=dict(title="Stress"),
            cmid=0
        ),
        name="Stress"
    ))




@app.callback(
    Output("stress_plot", "figure"),
    Output("summary", "children"),
    Output("status_box", "children"),
    Input("run_btn", "n_clicks"),
    State("Beam_Length", "value"),
    State("Case", "value"),
    State("E", "value"),
    State("rho", "value"),
    State("Distributed_Load", "value"),
    State("Gravity", "value"),
    # Rect inputs
    State("H_tip", "value"),
    State("H_fix", "value"),
    State("W_tip", "value"),
    State("W_fix", "value"),
    State("h_tip", "value"),
    State("h_fix", "value"),
    State("w_tip", "value"),
    State("w_fix", "value"),
    # Circ inputs
    State("D_tip", "value"),
    State("D_fix", "value"),
    State("d_tip", "value"),
    State("d_fix", "value"),
    # Loads text
    State("loads_text", "value"),
    State("moments_text", "value"),
)
def run_sim(
    n_clicks,
    Beam_Length, Case, E, rho, Distributed_Load, Gravity,
    H_tip, H_fix, W_tip, W_fix, h_tip, h_fix, w_tip, w_fix,
    D_tip, D_fix, d_tip, d_fix,
    loads_text, moments_text
):
    if n_clicks == 0:
        fig = go.Figure()
        fig.update_layout(title="Click 'Solve'")
        return fig, "", ""

    try:
        Concentrated_loadings = parse_csv_pairs(loads_text)
        Concentrated_moments = parse_csv_pairs(moments_text)

        CS_Geometry = np.array([
            Beam_Length,
            H_tip, H_fix,
            W_tip, W_fix,
            h_tip, h_fix,
            w_tip, w_fix,
            D_tip, d_tip,
            D_fix, d_fix
        ], dtype=float)

        Domain, ds, Theta, M, I, Y_Range, status_log  = solve_beam(
            Beam_Length=float(Beam_Length),
            Case=int(Case),
            CS_Geometry=CS_Geometry,
            E=float(E),
            rho=float(rho),
            Concentrated_loadings=Concentrated_loadings,
            Concentrated_moments=Concentrated_moments,
            Distributed_Load=float(Distributed_Load),
            Gravity=float(Gravity),
            Number_of_divisions=1000
        )

        x, y, Px, Ph, Stress = postprocess(Domain, ds, Theta, M, I, Y_Range, DP=51)

        # Plotly contour (stress field)
        fig = go.Figure()

        curved_stress_scatter(fig, Px+Beam_Length, Ph, Stress, downsample=1)

        
        # Deformed centerline
        fig.add_trace(go.Scatter(
            x=x+Beam_Length, y=y,
            mode="lines",
            name="Deformed centerline",
            line=dict(width=3, color="black")
        ))
        

        # Undeformed axis: Domain vs 0
        fig.add_trace(go.Scatter(
            x=Domain, y=np.zeros_like(Domain),
            mode="lines",
            name="Undeformed axis",
            line=dict(width=2, dash="dash", color="gray")
        ))

        fig.update_layout(
            title="Stress distribution (deformed coordinates)",
            xaxis_title="x",
            yaxis_title="y",
            height=650,
            legend=dict(orientation="h")
        )

        # status = "Done."
        summary = html.Div([
            html.Div(f"Tip deformation (dx0, dy0) = ({(x[0]+Beam_Length):.6f}, {y[0]:.6f})"),
            html.Div(f"Max |Stress| (excluding first station) ≈ {np.nanmax(np.abs(Stress[:,1:])):.6g}")
        ])
        
        status_text = "\n".join(status_log)

        return fig, summary, status_text

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error")
        return fig, f"Error: {e}", ""


#if __name__ == "__main__":
#    app.run(debug=True)
    
if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=True)
    

