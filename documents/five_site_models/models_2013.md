# Five offshore-site statistical models (verified 2013 precursor)

Source: Li, Gao and Moan, OMAE2013-10156, *Joint Environmental Data at Five European Offshore Sites for Design of Combined Wind and Wave Energy Devices*. Equations (4)-(20), Appendix A Tables 3-5, page 9.

Public source record: https://nva.sikt.no/registration/0198cc75d2b3-c042eb68-a2c9-4568-b511-d5f6e3ce904a

Requested journal article: https://doi.org/10.1115/1.4029842 (2015). Its repository record says the article is unavailable due to copyright restrictions: https://nva.sikt.no/registration/0198cc5cc8c7-d63d01f6-04e6-4863-b336-ab103dd76c5d

**These are verified conference-paper coefficients, not verified 2015 journal coefficients.** The author's thesis identifies the journal article as an extension of this conference paper. Do not silently substitute these coefficients when reproducing the journal article.

## Sites and variables

| Site | Name | Region | Actual depth in source (m) |
|---|---|---|---:|
| 01 | Sem Rev (SEM-REV) | Atlantic | 33 |
| 03 | Buoy Cabo Silleiro | Atlantic | 449 |
| 05 | Wave Hub | Atlantic | 43 |
| 14 | Norway 5 | North Sea | 202 |
| 15 | North Sea Center | North Sea | 29 |

Depths above are geographic depths in Table 1, not the generic depths used for concept studies.

- U: one-hour mean wind speed at 10 m above sea level, in m/s.
- H: significant wave height, in m.
- T: wave spectral peak period, in s.
- Database: simultaneous hourly hindcast data from 2001-2010.
- ln is the natural logarithm. Insert numerical values in the stated units into the empirical power functions.

## Full trivariate model

The density factorization is

$$f_{U,H,T}(u,h,t)=f_U(u)f_{H|U}(h|u)f_{T|U,H}(t|u,h).$$

Define Weibull(shape alpha, scale beta) by

$$F_W(x)=1-\exp[-(x/\beta)^\alpha],\quad x\geq0.$$

The first two factors are

$$U\sim\operatorname{Weibull}(\alpha_U,\beta_U),$$

$$H|U=u\sim\operatorname{Weibull}(a_1+a_2u^{a_3},\;b_1+b_2u^{b_3}).$$

The wave-period factor is lognormal:

$$\ln T|U=u,H=h\sim N(\mu_{\ln T}(u,h),\sigma_{\ln T}^2(h)).$$

Compute the arithmetic mean m and coefficient of variation v first:

$$\bar T(h)=e_1+e_2h^{e_3},\qquad \bar u(h)=f_1+f_2h^{f_3},$$

$$m(u,h)=\bar T(h)\left[1+\theta\left(\frac{u-\bar u(h)}{\bar u(h)}\right)^\gamma\right],$$

$$v(h)=k_1+k_2\exp(k_3h).$$

Then

$$\sigma_{\ln T}^2(h)=\ln[1+v(h)^2],\qquad \mu_{\ln T}(u,h)=\ln m(u,h)-\tfrac12\ln[1+v(h)^2].$$

All five fitted gamma values equal 1. The e coefficients describe arithmetic mean period; they are not lognormal log-space means. The k formula describes coefficient of variation, not variance.

### Full-model coefficients (Table 5)

| Parameter | 01 | 03 | 05 | 14 | 15 |
|---|---:|---:|---:|---:|---:|
| alpha_U | 2.262 | 2.002 | 2.050 | 2.029 | 2.299 |
| beta_U | 7.635 | 7.866 | 7.859 | 9.409 | 8.920 |
| a1 | 1.894 | 1.643 | 2.044 | 2.136 | 1.755 |
| a2 | 0.012 | 0.093 | 0.034 | 0.013 | 0.184 |
| a3 | 1.741 | 1.000 | 1.375 | 1.709 | 1.000 |
| b1 | 0.929 | 1.969 | 1.323 | 1.816 | 0.534 |
| b2 | 0.024 | 0.031 | 0.032 | 0.024 | 0.070 |
| b3 | 1.827 | 1.644 | 1.757 | 1.787 | 1.435 |
| theta | -0.268 | -0.143 | -0.233 | -0.255 | -0.477 |
| gamma | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| e1 | 5.0 | 5.0 | 8.0 | 8.0 | 5.563 |
| e2 | 5.883 | 5.970 | 2.600 | 1.938 | 0.798 |
| e3 | 0.201 | 0.223 | 0.409 | 0.486 | 1.0 |
| f1 | 2.0 | 1.0 | 1.8 | 2.5 | 3.5 |
| f2 | 3.947 | 4.055 | 3.478 | 3.001 | 3.592 |
| f3 | 0.620 | 0.466 | 0.667 | 0.745 | 0.735 |
| k1 | -0.002 | 0.030 | 0.002 | -0.001 | 0.050 |
| k2 | 0.341 | 0.234 | 0.298 | 0.316 | 0.388 |
| k3 | -0.186 | -0.221 | -0.166 | -0.145 | -0.321 |

## Separate wave-only model and simplified trivariate model

The wave-only model uses

$$f_{H,T}(h,t)=f_H(h)f_{T|H}(t|h),$$

where f_H is a hybrid lognormal/Weibull (Lonowe) density:

$$f_H(h)=\begin{cases}
\dfrac{1}{h\sigma_{LHM}\sqrt{2\pi}}\exp\left[-\dfrac{(\ln h-\mu_{LHM})^2}{2\sigma_{LHM}^2}\right],&0<h\leq h_0,\\
\dfrac{\alpha_{HM}}{\beta_{HM}}(h/\beta_{HM})^{\alpha_{HM}-1}\exp[-(h/\beta_{HM})^{\alpha_{HM}}],&h>h_0.
\end{cases}$$

The source chooses the Weibull tail parameters to match both PDF and CDF at h0; printed rounded coefficients may leave small mismatches. This is a piecewise density, not a weighted mixture of two complete distributions.

The period distribution is

$$\ln T|H=h\sim N(c_1+c_2h^{c_3},\;d_1+d_2\exp(d_3h)).$$

The d expression is **log-space variance**, not standard deviation. The simplified trivariate model uses this same period factor with the full model's U and H|U factors:

$$f_{U,H,T}(u,h,t)\approx f_U(u)f_{H|U}(h|u)f_{T|H}(t|h).$$

The separate fitted marginal H distribution need not equal the marginal obtained by integrating the trivariate fit over U.

### Wave-only coefficients (Table 4)

| Parameter | 01 | 03 | 05 | 14 | 15 |
|---|---:|---:|---:|---:|---:|
| h0 | 3.5 | 5.0 | 5.0 | 5.0 | 2.5 |
| mu_LHM | 0.256 | 0.783 | 0.595 | 0.871 | 0.334 |
| sigma_LHM | 0.583 | 0.493 | 0.557 | 0.506 | 0.615 |
| alpha_HM | 1.160 | 1.385 | 1.179 | 1.433 | 1.369 |
| beta_HM | 1.309 | 2.229 | 1.785 | 2.547 | 1.653 |
| c1 | 1.900 | 2.008 | 2.004 | 1.886 | 1.587 |
| c2 | 0.429 | 0.363 | 0.321 | 0.365 | 0.222 |
| c3 | 0.272 | 0.295 | 0.332 | 0.312 | 0.674 |
| d1 | 0.001 | 0.001 | 0.001 | 0.001 | 0.008 |
| d2 | 0.205 | 0.068 | 0.103 | 0.105 | 0.227 |
| d3 | -0.487 | -0.300 | -0.285 | -0.264 | -0.956 |

## Interpretation and implementation limits

The model describes long-term environmental states, not response maxima or a time-series dependence model. The source assumes collinear wind and waves; direction distributions are excluded. T refers to the complete spectrum, including wind sea and swell.

The full conditional mean-period formula with negative theta can produce nonpositive m at sufficiently large u relative to h. It is an empirical fit and must not be extrapolated blindly over the entire positive quadrant. Check m>0 and v>0 before evaluating the lognormal distribution; do not silently clip invalid values and call the result the published model.

Verification performed: rendered and visually checked Table 3-5 on page 9 and the full conditional-model equations on page 6 against the text extraction.
