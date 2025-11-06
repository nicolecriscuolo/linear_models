linear_models
================
Nicole Criscuolo
2025-11-06

Load NYC airbnb data.

``` r
data(nyc_airbnb)
```

Look at data/ do some cleaning.

``` r
nyc_airbnb = 
  nyc_airbnb |> 
  mutate(
    stars = review_scores_location / 2
  ) |> 
  rename(borough = neighbourhood_group) |> 
  filter(borough != "Staten Island") |> 
  select(price, stars, borough, room_type, neighbourhood)
```

Do regression!!!

``` r
fit = lm(price ~ stars + borough, data = nyc_airbnb)
```

Do some additional cleaning then refit.

``` r
nyc_airbnb =
  nyc_airbnb |> 
  mutate(
    borough = fct_infreq(borough),
    room_type = fct_infreq(room_type))

fit = lm(price ~ stars + borough, data = nyc_airbnb)
```

Look at `lm` stuff.

``` r
summary(fit)
names(summary(fit))
summary(fit)[["coefficients"]]
summary(fit)[["df"]]

fitted.values(fit)
```

``` r
fit |> 
  broom::tidy() |> 
  mutate(
    term = str_replace(term, "borough", "Borough: ")
  ) |> 
  select(term, estimate, p.value) |> 
  knitr::kable(digits = 3)
```

| term              | estimate | p.value |
|:------------------|---------:|--------:|
| (Intercept)       |   19.839 |   0.104 |
| stars             |   31.990 |   0.000 |
| Borough: Brooklyn |  -49.754 |   0.000 |
| Borough: Queens   |  -77.048 |   0.000 |
| Borough: Bronx    |  -90.254 |   0.000 |

``` r
fit |> 
  broom::glance()
```

    ## # A tibble: 1 × 12
    ##   r.squared adj.r.squared sigma statistic   p.value    df   logLik    AIC    BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>     <dbl> <dbl>    <dbl>  <dbl>  <dbl>
    ## 1    0.0342        0.0341  182.      271. 6.73e-229     4 -202113. 4.04e5 4.04e5
    ## # ℹ 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>
