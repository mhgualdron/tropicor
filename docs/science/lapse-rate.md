# Physical Lapse Rates & Orographic Downscaling

Vertical temperature downscaling in mountainous terrain is governed by the thermodynamics of the planetary boundary layer and the free troposphere. This document details the mathematical framework, empirical benchmark findings, statistical properties, and fundamental physical limitations of 1D lapse-rate adjustments.

---

## Thermodynamic Formulation

As air parcels ascend in the atmosphere, they expand due to lower ambient pressure and cool adiabatically. The rate at which atmospheric temperature decreases with increasing altitude $z$ is termed the **environmental lapse rate**:

$$\Gamma = -\frac{\partial T}{\partial z}$$

In hydrostatic equilibrium and well-mixed convective conditions:
- **Dry Adiabatic Lapse Rate ($\Gamma_d$)**:
  $$\Gamma_d = \frac{g}{c_p} \approx 0.0098\,^\circ\text{C}/\text{m} \quad (9.8\,^\circ\text{C}/\text{km})$$
  where $g = 9.80665\,\text{m}/\text{s}^2$ is gravitational acceleration and $c_p = 1004.67\,\text{J}/(\text{kg}\cdot\text{K})$ is the specific heat capacity of dry air at constant pressure.
- **Moist (Saturated) Adiabatic Lapse Rate ($\Gamma_m$)**:
  $$\Gamma_m = \Gamma_d \left( \frac{1 + \frac{L_v r_s}{R_d T}}{1 + \frac{L_v^2 r_s}{c_p R_v T^2}} \right) \approx 0.004 \text{ to } 0.006\,^\circ\text{C}/\text{m}$$
  Latent heat release during water vapor condensation reduces the net cooling rate.
- **Standard Environmental Lapse Rate ($\Gamma_{\text{std}}$)**:
  $$\Gamma_{\text{std}} = 0.0065\,^\circ\text{C}/\text{m} \quad (6.5\,^\circ\text{C}/\text{km})$$
  Adopted as the international standard atmosphere (ICAO/ISA) representative of the tropical middle and lower troposphere.

---

## Topographic Elevation Discrepancy

In global reanalyses such as ERA5, surface elevation is derived from truncated spherical harmonics and averaged over grid boxes of approximately $31\,\text{km} \times 31\,\text{km}$. In steep tropical terrain like the Colombian Andes, this spatial filtering leads to elevation discrepancies:

$$\Delta z = z_{\text{station}} - z_{\text{model}}$$

The vertical temperature correction is defined as:

$$T_{\text{corrected}} = T_{\text{raw}} - \Gamma \cdot \Delta z = T_{\text{raw}} - \Gamma \cdot (z_{\text{station}} - z_{\text{model}})$$

### The Bucaramanga UIS Benchmark Case

The benchmark station evaluated in the research is the **Universidad Industrial de Santander (UIS)** station in Bucaramanga (Station Code `23195040`, Santander Department):

- **Station Altitude ($z_{\text{station}}$)**: $898.0\,\text{m}$ masl (located on the plateau valley floor).
- **ERA5 Grid Elevation ($z_{\text{ERA5}}$)**: $2,118.0\,\text{m}$ masl (smoothed with the high summits of the adjacent Cordillera Oriental).
- **Elevation Offset ($\Delta z$)**: $898.0 - 2,118.0 = -1,220.0\,\text{m}$.

Applying the standard environmental lapse rate ($\Gamma = 0.0065\,^\circ\text{C}/\text{m}$):

$$\Delta T = -0.0065 \cdot (-1,220.0) = +7.93\,^\circ\text{C}$$

Without vertical correction, raw ERA5 reports temperatures that are systematically $\sim 8^\circ\text{C}$ too cold, severely distorting thermal amplitudes and surface energy balance calculations.

---

## Mathematical Properties & Statistical Invariance

### Shift Invariance of Pearson Correlation

A crucial statistical property of constant lapse-rate adjustment is that it represents an affine transformation ($Y = X + c$, where $c = -\Gamma \cdot \Delta z$ is a scalar constant):

$$r(T_{\text{raw}} + c, T_{\text{obs}}) = \frac{\text{Cov}(T_{\text{raw}} + c, T_{\text{obs}})}{\sigma_{T_{\text{raw}} + c} \cdot \sigma_{T_{\text{obs}}}} = \frac{\text{Cov}(T_{\text{raw}}, T_{\text{obs}})}{\sigma_{T_{\text{raw}}} \cdot \sigma_{T_{\text{obs}}}} = r(T_{\text{raw}}, T_{\text{obs}})$$

The Pearson correlation coefficient $r$ is **mathematically invariant** under constant lapse-rate correction. 

### Why Correlation Alone Is Insufficient

Because Pearson correlation measures linear co-movement rather than absolute scale or position, raw ERA5 can exhibit a high correlation ($r > 0.85$) while suffering from an unacceptable $8^\circ\text{C}$ cold bias.

The vertical lapse-rate correction directly resolves this systemic discrepancy:
- **Mean Bias**: Reduces toward $\sim 0\,^\circ\text{C}$.
- **RMSE & MAE**: Drop significantly (e.g., from $>8^\circ\text{C}$ to $<1.5^\circ\text{C}$).
- **Euclidean Distance ($d$)**: Greatly contracts between multi-year climatological annual cycles.

---

## Physical Limitations & Scientific Scope

> [!WARNING]
> **Boundary Layer Assumptions & Limitations**:
> Constant lapse-rate correction is a **first-order free-tropospheric physical model**. Users and researchers must not overstate its capabilities, as it possesses fundamental physical boundaries:

1. **Nocturnal Thermal Inversions**:
   In complex Andean topography, radiative cooling during clear nights causes dense, cold air to drain downslope (katabatic drainage) and pool at the bottom of intramontane valleys (**cold-air pooling**). In these nocturnal boundary layers, the effective lapse rate frequently flattens to zero or reverses ($\Gamma \le 0$, where temperature increases with height).
2. **Differential Bias Between Extrema ($T_{\max}$ vs $T_{\min}$)**:
   Because daytime solar heating generates convective mixing that aligns closely with adiabatic lapse rates, a constant $\Gamma = 0.0065\,^\circ\text{C}/\text{m}$ successfully corrects daytime maximum ($T_{\max}$) and mean temperatures. However, applying this uniform warming to nighttime minimum temperatures ($T_{\min}$) frequently leads to over-correction.
3. **Correlation Breakdown in $T_{\min}$**:
   This nocturnal boundary-layer decoupling explains why empirical evaluations across Colombia's Andean stations reveal a collapse in ERA5 minimum temperature correlation (median $r \approx 0.30$).

### Bridge to v0.3.0 Machine Learning Downscaling

These physical limitations demonstrate that a linear, one-dimensional lapse rate cannot capture complex microclimatic phenomena such as nocturnal cold-air trapping, valley ventilation, and aspect-driven insolation differences.

This boundary serves as the formal scientific justification for **v0.3.0 Topographic Machine Learning Downscaling**, where non-linear regressors learn microclimate dynamics by conditioning on high-resolution DEM derivatives (slope, aspect, terrain roughness, and valley depth indices).
