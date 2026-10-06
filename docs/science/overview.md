# Academic Heritage & Orographic Bias

## Research Origins

TROPICOR originates from undergraduate geological and atmospheric research conducted at **Universidad Nacional de Colombia** (Bogotá), directed by:

- **PhD Germán Andrés Prieto Gómez** (Department of Geosciences)
- **PhD Daniel Hernández Deckers** (Department of Geosciences)
- **Lead Author & Architect**: Mateo Hernández Gualdrón

The investigation conducted a comprehensive 30-year empirical evaluation (1990–2019) comparing the European Centre for Medium-Range Weather Forecasts (**ECMWF**) ERA5 global reanalysis against in-situ station records from the **Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)** across Colombia's 6 natural regions.

---

## The Orographic Bias Phenomenon

In global reanalyses, surface elevation is averaged over a $\sim 30\,\text{km} \times 30\,\text{km}$ grid box. In regions with steep topographic relief—such as the three branches of the Colombian Andes—this spatial smoothing introduces massive discrepancies between model elevation ($z_{\text{ERA5}}$) and ground station elevation ($z_{\text{station}}$):

$$\Delta z = z_{\text{station}} - z_{\text{ERA5}}$$

### The Bucaramanga Benchmark Case

Consider the **Universidad Industrial de Santander (UIS)** station in Bucaramanga (Station Code: `23195040`):

- Station Altitude ($z_{\text{station}}$): **$898\,\text{m}$ masl** (located in the valley plateau)
- ERA5 Grid Elevation ($z_{\text{ERA5}}$): **$2,118\,\text{m}$ masl** (smoothed with the adjacent Cordillera Oriental summit)
- Elevation Offset ($\Delta z$): **$-1,220\,\text{m}$**

Because temperatures decrease with height in the troposphere according to the environmental lapse rate:

$$\Gamma \approx 0.0065\,^\circ\text{C}/\text{m} \quad (6.5\,^\circ\text{C}/\text{km})$$

A vertical offset of $\Delta z = -1,220\,\text{m}$ produces an uncorrected temperature error of:

$$\Delta T = -\Gamma \cdot \Delta z = -(0.0065) \cdot (-1220) \approx +7.93\,^\circ\text{C}$$

Without orographic correction, raw ERA5 reports temperatures that are nearly **$8^\circ\text{C}$ too cold** at the UIS station, completely distorting local thermal amplitudes and seasonal regimes.

---

## Regional Disparities

The research demonstrated that ERA5 reliability is fundamentally determined by terrain complexity:

1. **Lowland Plains (Orinoquía, Caribe, Amazonía)**: Minimal elevation offset ($\Delta z \approx 0$). ERA5 demonstrates high Pearson correlation ($r > 0.85$) and low RMSE.
2. **Andean Valleys and Slopes**: Severe elevation offsets ($\Delta z$ up to $\pm 1,500\,\text{m}$). ERA5 fails critically unless orographic vertical correction and statistical downscaling are applied.

TROPICOR operationalizes these empirical findings into an automated, production-ready Python package.
