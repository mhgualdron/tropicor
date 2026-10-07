# Scientific Verification Metrics

Evaluating climate reanalyses against in-situ station observations requires a multi-criteria statistical framework. A single metric (such as correlation $r$) can mask systemic elevation-driven biases or variance suppression.

TROPICOR implements the standardized `compute_validation_metrics` function returning an immutable `ValidationReport`.

---

## 1. Metric Formulations

Given synchronized observed time series $y_{\text{obs}} = \{o_1, \dots, o_n\}$ and modeled reanalysis series $y_{\text{mod}} = \{m_1, \dots, m_n\}$ across $n$ valid time steps:

### Pearson Correlation ($r$) and Two-Tailed Significance ($p$)
Measures temporal phase coherence and linear association:

$$r = \frac{\sum_{i=1}^n (o_i - \bar{o})(m_i - \bar{m})}{\sqrt{\sum_{i=1}^n (o_i - \bar{o})^2} \sqrt{\sum_{i=1}^n (m_i - \bar{m})^2}}$$

Where $p$ is calculated from the Student's $t$-distribution with $n - 2$ degrees of freedom.

---

### Root Mean Square Error ($\text{RMSE}$)
Penalizes large outlier discrepancies quadratically:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (m_i - o_i)^2}$$

---

### Mean Absolute Error ($\text{MAE}$)
Measures the average magnitude of absolute errors:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |m_i - o_i|$$

---

### Mean Bias ($\text{Bias}$)
Quantifies average over- or under-estimation:

$$\text{Bias} = \bar{m} - \bar{o} = \frac{1}{n} \sum_{i=1}^n (m_i - o_i)$$

In tropical mountainous terrain, uncorrected ERA5 temperature exhibits systematic negative bias ($\text{Bias} \ll 0$) due to unresolved mountain peaks.

---

### Percent Bias ($\text{PBIAS}$)
Expresses the average tendency of simulated values to be larger or smaller than their observed counterparts as a percentage:

$$\text{PBIAS} = 100 \times \frac{\sum_{i=1}^n (m_i - o_i)}{\sum_{i=1}^n o_i}$$

---

### Kling-Gupta Efficiency ($\text{KGE}$)
Proposed by Gupta et al. (2009), KGE decomposes goodness-of-fit into three orthogonal components of hydrological and climatological performance:

$$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$

Where:
- **Correlation ratio**: $r$ (temporal dynamic agreement)
- **Variability ratio**: $\alpha = \frac{\sigma_{\text{mod}}}{\sigma_{\text{obs}}}$ (relative dispersion)
- **Bias ratio**: $\beta = \frac{\mu_{\text{mod}}}{\mu_{\text{obs}}}$ (relative volume/mean)

An ideal simulation yields $\text{KGE} = 1.0$. Typically, $\text{KGE} > 0.5$ denotes satisfactory performance, while $\text{KGE} < 0$ implies the mean observed value is a better predictor than the model.

---

### Euclidean Distance ($d$)
Calculates the geometric metric distance between series:

$$d = \sqrt{\sum_{i=1}^n (m_i - o_i)^2}$$

---

## 2. Tropical Robustness & Numerical Guards

Tropical climatology presents unique edge cases:
1. **Hyper-arid seasons**: In regions like the Guajira peninsula or during severe drought, monthly precipitation may be near zero ($\mu_{\text{obs}} \approx 0$). In this regime, the ratio $\beta = \mu_{\text{mod}} / \mu_{\text{obs}}$ diverges to infinity.
2. **Zero-variance series**: During extended droughts or sensor pegging, $\sigma_{\text{obs}} \approx 0$, which causes $\alpha = \sigma_{\text{mod}} / \sigma_{\text{obs}}$ to divide by zero.

TROPICOR guards all divisions using a tolerance parameter $\epsilon = 10^{-6}$:
- If $|\mu_{\text{obs}}| < \epsilon$ or $\sigma_{\text{obs}} < \epsilon$, a `UserWarning` is issued and `kge = np.nan`.
- **Fault-Tolerant Reporting**: The calculation of other independent metrics ($\text{RMSE}$, $\text{MAE}$, $\text{Bias}$, $d$) is never aborted, ensuring pipeline pipelines remain robust.

---

## 3. Example Usage

```python
import pandas as pd
from tropicor.core.metrics import compute_validation_metrics

# Load or generate observed and modeled series
obs = pd.Series(
    [22.1, 22.4, 21.9, 23.0, 22.8],
    index=pd.date_range("2010-01-01", periods=5, freq="MS"),
)
mod = pd.Series(
    [14.2, 14.5, 13.9, 15.1, 14.7],
    index=pd.date_range("2010-01-01", periods=5, freq="MS"),
)

report = compute_validation_metrics(obs, mod)

print(f"Pearson r: {report.pearson_r:.3f}")
print(f"Mean Bias: {report.bias:.2f} °C")
print(f"KGE:       {report.kge:.3f}")
print(f"RMSE:      {report.rmse:.2f} °C")

# Export as dictionary or pandas Series
summary = report.to_dict()
```
