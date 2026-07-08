# -*- coding: utf-8 -*-
"""
================================================================================
 modelo_milp_compra.py
 Modelo de optimización de compra de productos importados — IKEA Food Chile
 Tesis Magíster en Organización y Liderazgo (MOL · UNAB)
 Francisco González — "Diseño de un Modelo de Compra de Productos Importados
 en IKEA Food Chile"
================================================================================

Este módulo implementa el MODELO MILP descrito en el Capítulo 4.4 y operado en
el Capítulo 9 de la tesis. La formulación es una adaptación de Pauls-Worm et al.
(2014) con las tres extensiones propias del caso IKEA Food: (i) pedido en pallets
completos, (ii) disyunción de capacidad de contenedor (23 o 40–44 pallets) y
(iii) vida útil residual al momento del arribo.

El MILP consume los INSUMOS generados por los modelos intermedios (Capítulo 6):
    - D̂_{p,t}  : forecast de demanda. Se corrige por el SESGO sistemático de
                 +23 % detectado al lag operativo de 6 meses (sección 8).
    - V_{p,t}  : vencimientos proyectados del stock vigente (modelo de
                 obsolescencia + FEFO retroactivo, Ec. 13).
    - I_{p,0}  : inventario consolidado inicial (CD + tiendas).
    - A_{p,t}  : arribos ya confirmados en tránsito (proyección de inventario).
    - σ_p      : desviación estándar del error de pronóstico = RMSE (sección 8),
                 que parametriza el stock de seguridad SS = Z · σ · √L (Ec. 4).

------------------------------- FORMULACIÓN ------------------------------------
Conjuntos
    p ∈ P   productos (SKU importados)
    t ∈ {1..T}   períodos (meses); horizonte T = 12

Variables de decisión
    n_{p,t} ∈ ℤ₊      nº de pallets pedidos                         (Ec. 9)
    Q_{p,t} = n·CAP_p unidades pedidas                              (Ec. 9)
    I_{p,t} ≥ 0       inventario al cierre del período t
    S_{p,t} ≥ 0       quiebre (demanda no satisfecha, venta perdida)
    δ_{p,t} ∈ {0,1}   1 si se genera un pedido en t
    b^s_{p,t}, b^f_{p,t} ∈ {0,1}  selección de contenedor chico/lleno (Ec. 10)

Función objetivo                                                    (Ec. 7)
    min  Σ_{p,t} [ h_p·I_{p,t} + p_p·S_{p,t} + o·δ_{p,t} + c_u,p·Q_{p,t} ]
    (el término w_p·V_{p,t} es constante porque V es dato del modelo de
     obsolescencia; se reporta pero no altera el óptimo. Bajo CIP el flete NO
     entra en la función objetivo.)

Sujeto a
    Balance con vencimientos y quiebre (extensión de Ec. 8):
        I_{p,t} = I_{p,t-1} + A_{p,t} + Q_{p,t-L} − D_{p,t} + S_{p,t} − V_{p,t}
    Stock de seguridad (Ec. 5, σ = RMSE, Ec. 4):
        I_{p,t} ≥ SS_{p,k}            (soft por defecto para robustez)
    Integralidad de pallet (Ec. 9):
        Q_{p,t} = n_{p,t} · CAP_p
    Disyunción de contenedor (Ec. 10):
        n_{p,t} ∈ {0} ∪ [n_min, 23] ∪ [40, 44]
    Vida útil residual al arribo (Ec. 11):
        I_{p,t-1} + A_{p,t} + Q_{p,t-L} ≤ Σ_{k=0}^{VU_p-1} D̂_{p,t+k}

Solver: PuLP + CBC (código abierto), tal como se especifica en la sección 5.3.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pulp


# =============================================================================
# 1. PARÁMETROS QUE PROVIENEN DE LOS MODELOS INTERMEDIOS Y DEL ANÁLISIS (Cap. 8)
# =============================================================================

# Nivel de servicio diferenciado por clase ABC → factor Z de la normal estándar
#   A: 98 %  ·  B: 95 %  ·  C: 90 %   (Tabla 3 de la tesis)
Z_BY_CLASS: Dict[str, float] = {"A": 2.054, "B": 1.645, "C": 1.282}

# Sesgo sistemático del forecast nacional al lag operativo de 6 meses (sección 8).
#   El forecast sobreestima la demanda en +23,2 %  →  se corrige antes de optimizar.
DEFAULT_FORECAST_BIAS: float = 0.232

# Lead time completo de importación (orden → liberación SEREMI), en meses.
DEFAULT_LEAD_TIME: int = 6


def safety_stock(z: float, sigma: float, lead_time: int) -> float:
    """Stock de seguridad — Ec. (4):  SS = Z · σ · √L.

    σ es la desviación estándar del error de pronóstico, estimada por el RMSE
    del análisis de precisión por lag (sección 8). L es el lead time en períodos.
    """
    return z * sigma * math.sqrt(lead_time)


def debias_forecast(forecast: np.ndarray, bias: float = DEFAULT_FORECAST_BIAS) -> np.ndarray:
    """Corrige el sesgo sistemático del forecast nacional.

    Si el pronóstico sobreestima en una fracción `bias` (p. ej. +0,232),
    la demanda esperada insesgada es  D̂_insesgada = D̂_forecast / (1 + bias).
    Esta corrección es el mecanismo por el cual el hallazgo de la sección 8
    (sobre-pronóstico → sobrestock → vencimientos) se traduce en menos compra.
    """
    return np.asarray(forecast, dtype=float) / (1.0 + bias)


# =============================================================================
# 2. ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class SKU:
    """Parámetros de un producto importado (una fila de la tabla del Bloque 3)."""
    sku_id: str
    abc_class: str                  # "A" | "B" | "C"
    units_per_pallet: int           # CAP_p  (Ec. 9)
    shelf_life_months: int          # VU_p   (Ec. 11) — vida útil residual al arribo
    holding_cost: float             # h_p   ($/unidad/período) = tarifa bodega / CAP_p
    unit_cost: float                # c_u,p ($/unidad)  (Dynamics)
    stockout_penalty: float         # p_p   ($/unidad)  ≈ precio de venta
    expiry_cost: float              # w_p   ($/unidad)  costo medio al vencimiento
    rmse: float                     # σ_p   (= RMSE del forecast, sección 8)
    init_inventory: float = 0.0     # I_{p,0}

    def z(self) -> float:
        return Z_BY_CLASS[self.abc_class]


@dataclass
class ModelConfig:
    """Configuración global del MILP."""
    horizon: int = 12               # T (meses)
    lead_time: int = DEFAULT_LEAD_TIME   # L (meses)
    forecast_bias: float = DEFAULT_FORECAST_BIAS
    order_fixed_cost: float = 50_000.0   # o  ($/pedido)
    # Disyunción de contenedor (Ec. 10)
    small_container_min: int = 1     # n_min
    small_container_max: int = 23
    full_container_min: int = 40
    full_container_max: int = 44
    # Stock de seguridad como restricción dura o blanda (penalizada)
    hard_safety_stock: bool = False
    ss_shortfall_penalty: float = 1.0e6   # penalización por unidad bajo SS (modo blando)
    solver_msg: bool = False


# =============================================================================
# 3. CONSTRUCCIÓN Y RESOLUCIÓN DEL MILP
# =============================================================================

def solve_milp(
    skus: List[SKU],
    forecast: Dict[str, np.ndarray],          # D̂_{p,t} crudo (con sesgo), shape [T]
    expirations: Dict[str, np.ndarray],       # V_{p,t}  (modelo de obsolescencia)
    arrivals: Optional[Dict[str, np.ndarray]] = None,   # A_{p,t} arribos confirmados
    cfg: ModelConfig = ModelConfig(),
) -> Dict:
    """Construye y resuelve el modelo MILP de compra.

    Retorna un diccionario con el estado, el costo óptimo y un DataFrame de
    resultados por SKU y período (la tabla que la tesis escribe de vuelta a
    BigQuery para el área de compras).
    """
    T, L = cfg.horizon, cfg.lead_time
    periods = range(1, T + 1)

    prob = pulp.LpProblem("Compra_IKEA_Food_MILP", pulp.LpMinimize)

    # ---- variables ----
    n, Q, I, S, delta, b_s, b_f, ss_short = ({} for _ in range(8))
    for sk in skus:
        p = sk.sku_id
        for t in periods:
            n[p, t]     = pulp.LpVariable(f"n_{p}_{t}", lowBound=0, cat="Integer")
            Q[p, t]     = pulp.LpVariable(f"Q_{p}_{t}", lowBound=0, cat="Continuous")
            I[p, t]     = pulp.LpVariable(f"I_{p}_{t}", lowBound=0, cat="Continuous")
            S[p, t]     = pulp.LpVariable(f"S_{p}_{t}", lowBound=0, cat="Continuous")
            delta[p, t] = pulp.LpVariable(f"d_{p}_{t}", cat="Binary")
            b_s[p, t]   = pulp.LpVariable(f"bs_{p}_{t}", cat="Binary")
            b_f[p, t]   = pulp.LpVariable(f"bf_{p}_{t}", cat="Binary")
            ss_short[p, t] = pulp.LpVariable(f"sss_{p}_{t}", lowBound=0, cat="Continuous")

    # ---- demanda insesgada y stock de seguridad por SKU ----
    Dhat = {sk.sku_id: debias_forecast(forecast[sk.sku_id], cfg.forecast_bias) for sk in skus}
    SS   = {sk.sku_id: safety_stock(sk.z(), sk.rmse, L) for sk in skus}

    # ---- función objetivo (Ec. 7) ----
    obj = []
    for sk in skus:
        p = sk.sku_id
        for t in periods:
            obj += [
                sk.holding_cost * I[p, t],          # h · I
                sk.stockout_penalty * S[p, t],      # p · S
                cfg.order_fixed_cost * delta[p, t], # o · δ
                sk.unit_cost * Q[p, t],             # c_u · Q
            ]
            if not cfg.hard_safety_stock:
                obj.append(cfg.ss_shortfall_penalty * ss_short[p, t])
    prob += pulp.lpSum(obj)

    # ---- restricciones ----
    for sk in skus:
        p = sk.sku_id
        CAP, VU = sk.units_per_pallet, sk.shelf_life_months
        A = (arrivals or {}).get(p, np.zeros(T))
        for t in periods:
            inv_prev = sk.init_inventory if t == 1 else I[p, t - 1]
            arr_order = Q[p, t - L] if (t - L) >= 1 else 0   # pedido que arriba en t
            arr_conf  = float(A[t - 1])                       # arribo ya confirmado

            # Balance con vencimientos y quiebre (extensión de Ec. 8)
            prob += (
                I[p, t] == inv_prev + arr_conf + arr_order
                          - Dhat[p][t - 1] + S[p, t] - float(expirations[p][t - 1]),
                f"balance_{p}_{t}",
            )

            # Stock de seguridad (Ec. 5);  σ = RMSE (Ec. 4)
            if cfg.hard_safety_stock:
                prob += I[p, t] >= SS[p], f"ss_{p}_{t}"
            else:
                prob += I[p, t] + ss_short[p, t] >= SS[p], f"ss_{p}_{t}"

            # Integralidad de pallet (Ec. 9)
            prob += Q[p, t] == CAP * n[p, t], f"pallet_{p}_{t}"

            # Disyunción de contenedor (Ec. 10): n ∈ {0} ∪ [n_min,23] ∪ [40,44]
            prob += b_s[p, t] + b_f[p, t] <= 1, f"cont_sel_{p}_{t}"
            prob += n[p, t] <= cfg.small_container_max * b_s[p, t] \
                              + cfg.full_container_max * b_f[p, t], f"cont_ub_{p}_{t}"
            prob += n[p, t] >= cfg.small_container_min * b_s[p, t] \
                              + cfg.full_container_min * b_f[p, t], f"cont_lb_{p}_{t}"
            prob += delta[p, t] == b_s[p, t] + b_f[p, t], f"order_flag_{p}_{t}"

            # Vida útil residual al arribo (Ec. 11): el stock disponible tras el
            # arribo no puede superar la demanda consumible dentro de la vida útil.
            window = [Dhat[p][k - 1] for k in range(t, min(t + VU, T + 1))]
            prob += inv_prev + arr_conf + arr_order <= pulp.lpSum(window), f"shelf_{p}_{t}"

    # ---- resolver con CBC ----
    prob.solve(pulp.PULP_CBC_CMD(msg=cfg.solver_msg))
    status = pulp.LpStatus[prob.status]

    # ---- extraer resultados ----
    rows = []
    for sk in skus:
        p = sk.sku_id
        for t in periods:
            npal = int(round(n[p, t].value() or 0))
            qty  = (Q[p, t].value() or 0.0)
            cont = "—" if npal == 0 else ("Lleno (40-44)" if npal >= cfg.full_container_min
                                          else "Chico (≤23)")
            rows.append({
                "sku": p, "clase": sk.abc_class, "periodo": t,
                "pallets": npal, "unidades_pedidas": round(qty),
                "contenedor": cont,
                "inventario_fin": round(I[p, t].value() or 0.0),
                "stock_seguridad": round(SS[p]),
                "quiebre_u": round(S[p, t].value() or 0.0),
                "vencimiento_u": round(float(expirations[p][t - 1])),
                "demanda_insesgada": round(Dhat[p][t - 1]),
            })
    df = pd.DataFrame(rows)

    return {
        "status": status,
        "costo_total": pulp.value(prob.objective),
        "resultados": df,
        "safety_stock": SS,
        "demanda_insesgada": Dhat,
        "config": cfg,
        "skus": {s.sku_id: s for s in skus},
    }


# =============================================================================
# 4. VISUALIZACIÓN DE RESULTADOS (gráfico del modelo por SKU)
# =============================================================================

def plot_sku(result: Dict, sku_id: str, path: Optional[str] = None):
    """Grafica la trayectoria de inventario, el piso de stock de seguridad,
    los pedidos (arribos) y los vencimientos para un SKU."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try: font_manager.fontManager.addfont(fp)
        except Exception: pass
    plt.rcParams["font.family"] = "Liberation Sans"
    plt.rcParams["axes.unicode_minus"] = False
    NAVY, IKEA, YELL, RED = "#1F3864", "#0058A3", "#FFCC00", "#C0392B"

    df = result["resultados"]
    d = df[df["sku"] == sku_id].sort_values("periodo")
    cfg = result["config"]; L = cfg.lead_time
    t = d["periodo"].values

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    ax1.plot(t, d["inventario_fin"], "-o", color=IKEA, lw=2.2, ms=5,
             mfc="white", mec=IKEA, mew=1.5, label="Inventario proyectado", zorder=4)
    ax1.axhline(d["stock_seguridad"].iloc[0], color=RED, ls="--", lw=1.4,
                label=f"Stock de seguridad (Z·σ·√L = {d['stock_seguridad'].iloc[0]:,.0f})")

    # arribos = pedidos desplazados por el lead time
    arr = np.zeros(len(t))
    for _, r in d.iterrows():
        ta = int(r["periodo"]) + L
        if 1 <= ta <= len(t):
            arr[ta - 1] += r["unidades_pedidas"]
    ax1.bar(t, arr, width=0.55, color=YELL, ec=NAVY, lw=0.6, alpha=0.85,
            label=f"Arribos (pedidos, L={L})", zorder=2)

    venc = d["vencimiento_u"].values
    if venc.sum() > 0:
        ax1.bar(t, -venc, width=0.55, color=RED, ec="#7B241C", lw=0.6, alpha=0.7,
                label="Vencimientos", zorder=2)

    ax1.axhline(0, color=NAVY, lw=0.8)
    ax1.set_xlabel("Período (mes del horizonte)", color=NAVY)
    ax1.set_ylabel("Unidades", color=NAVY)
    ax1.set_xticks(t)
    ax1.grid(axis="y", color="#E8E8E8", lw=0.8); ax1.set_axisbelow(True)
    for s in ["top", "right"]: ax1.spines[s].set_visible(False)
    cls = result["skus"][sku_id].abc_class
    ax1.set_title(f"Modelo MILP — SKU {sku_id} (clase {cls}): plan de compra y trayectoria de inventario",
                  color=NAVY, fontweight="bold", loc="left", pad=10)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


# =============================================================================
# 5. CARGA DE DATOS DESDE BIGQUERY (esqueleto para producción)
# =============================================================================

def load_from_bigquery(project: str, dataset: str) -> Dict:
    """Esqueleto de integración con BigQuery (sección 5.3).

    En producción, reemplaza al generador sintético del demo. Cada consulta
    mapea a una tabla de los modelos intermedios del Capítulo 6:
        - parámetros SKU + clase ABC + CAP + vida útil + costos  → Bloque 3
        - forecast D̂_{p,t}                                       → tabla de forecast
        - vencimientos V_{p,t}                                    → vencimiento_producto_mes
        - inventario inicial / arribos confirmados               → fact_inventory_projection
        - σ_p (RMSE por SKU)                                      → fcst_accuracy_*_lag
    """
    raise NotImplementedError(
        "Conectar con google.cloud.bigquery y poblar SKU/forecast/expirations/"
        "arrivals desde las tablas del Capítulo 6. Ver docstring."
    )


# =============================================================================
# 6. DEMOSTRACIÓN EJECUTABLE (datos sintéticos con parámetros reales de la tesis)
# =============================================================================

def _demo_data(cfg: ModelConfig):
    """Genera un portafolio de 3 SKU (clases A, B, C) con estacionalidad y los
    parámetros de la tesis: sesgo +23 %, σ = RMSE, contenedores 23 / 40-44, etc."""
    rng = np.random.default_rng(42)
    T = cfg.horizon
    base = {"SKU-A-ALBONDIGA": (8000, "A", 1200, 8, 1567),
            "SKU-B-SALMON":    (3000, "B", 800, 10, 800),
            "SKU-C-MERMELADA": (1200, "C", 1500, 14, 363)}
    skus, forecast, expirations, arrivals = [], {}, {}, {}
    for sid, (mu, cls, cap, vu, rmse) in base.items():
        # demanda real con estacionalidad + ruido
        season = 1 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, T))
        real = mu * season * (1 + rng.normal(0, 0.05, T))
        # el forecast nacional SOBREESTIMA en +23 % (sesgo de la sección 8)
        fc = real * (1 + cfg.forecast_bias)
        forecast[sid] = fc
        # vencimientos proyectados del stock vigente (modelo de obsolescencia)
        expirations[sid] = np.clip(rng.normal(0.02 * mu, 0.01 * mu, T), 0, None)
        # arribos confirmados en tránsito en los primeros meses
        arr = np.zeros(T); arr[:cfg.lead_time] = mu  # ~1 mes de cobertura ya en camino
        arrivals[sid] = arr
        skus.append(SKU(
            sku_id=sid, abc_class=cls, units_per_pallet=cap, shelf_life_months=vu,
            holding_cost=120_000 / cap / 30,        # $120k/pallet/mes ÷ CAP ÷ 30 días
            unit_cost=3500, stockout_penalty=9000, expiry_cost=3500,
            rmse=rmse, init_inventory=1.5 * mu,
        ))
    return skus, forecast, expirations, arrivals


def main():
    cfg = ModelConfig(horizon=12, lead_time=6, forecast_bias=0.232,
                      hard_safety_stock=False, solver_msg=False)
    skus, forecast, expirations, arrivals = _demo_data(cfg)

    res = solve_milp(skus, forecast, expirations, arrivals, cfg)

    print("=" * 78)
    print(" MODELO MILP DE COMPRA — IKEA Food Chile  (demo con parámetros de tesis)")
    print("=" * 78)
    print(f" Estado del solver : {res['status']}")
    print(f" Costo total óptimo: ${res['costo_total']:,.0f} CLP")
    print(f" Sesgo corregido   : +{cfg.forecast_bias:.1%}  |  Lead time L = {cfg.lead_time} meses")
    print("\n Stock de seguridad por SKU (Ec. 4  SS = Z·σ·√L):")
    for sid, ss in res["safety_stock"].items():
        sk = res["skus"][sid]
        print(f"   {sid:18s} clase {sk.abc_class}  Z={sk.z():.3f}  σ={sk.rmse:>5.0f}  →  SS={ss:>8,.0f} u")

    df = res["resultados"]
    print("\n Plan de compra (períodos con pedido):")
    ped = df[df["pallets"] > 0][["sku", "periodo", "pallets", "unidades_pedidas",
                                 "contenedor", "inventario_fin", "stock_seguridad"]]
    print(ped.to_string(index=False) if len(ped) else "   (sin pedidos en el horizonte)")

    tot = df.groupby("sku").agg(
        pedidos=("pallets", lambda s: (s > 0).sum()),
        pallets_tot=("pallets", "sum"),
        quiebre_tot=("quiebre_u", "sum"),
        venc_tot=("vencimiento_u", "sum"),
    )
    print("\n Resumen por SKU:")
    print(tot.to_string())

    # gráfico del modelo para el SKU clase A
    fig = plot_sku(res, "SKU-A-ALBONDIGA", path="milp_plan_SKU-A.png")
    print("\n Figura guardada: milp_plan_SKU-A.png")
    return res


if __name__ == "__main__":
    main()
