cross_validation
================
Nicole Criscuolo
2025-11-11

``` r
data("lidar")
```

``` r
lidar_df =
  lidar |> 
  mutate(id = row_number())

lidar_df |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-2-1.png" width="90%" />

## Create dataframes

``` r
train_df = sample_frac(lidar_df, size = .8) |> 
  arrange(id)

test_df = anti_join(lidar_df, train_df, by = "id")
```

``` r
ggplot(train_df, aes(x = range, y = logratio)) +
  geom_point() +
  geom_point(data = test_df, color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-4-1.png" width="90%" />

Fit a few models to `train_df`.

``` r
linear_mod = lm(logratio ~ range, data = train_df)
```

Look at this. Not the best model; would be under fitting.

``` r
train_df |> 
  add_predictions(linear_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-6-1.png" width="90%" />

How to fit nonlinear models.

``` r
smooth_mod = mgcv::gam(logratio ~ s(range), data = train_df)

train_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-7-1.png" width="90%" />

This would be over fitting…

``` r
wiggly_mod = mgcv::gam(logratio ~ s(range, k = 30), sp = 10e-6, data = train_df)

train_df |> 
  add_predictions(wiggly_mod) |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-8-1.png" width="90%" />

Let’s see our root mean squared errors (RMSEs). Again, smooth model is
best but wiggly model not far off.

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.1264138

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.06591721

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.06731816

## Iterate

Takes lidar df and randomly splits 80/20 100 times.

``` r
cv_df =
  crossv_mc(lidar_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

Did this work? Yes!

``` r
cv_df |> pull(train) |> nth(1)
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   390  -0.0504     1
    ##  2   391  -0.0601     2
    ##  3   393  -0.0419     3
    ##  4   394  -0.0510     4
    ##  5   396  -0.0599     5
    ##  6   400  -0.0399     8
    ##  7   402  -0.0294     9
    ##  8   403  -0.0395    10
    ##  9   405  -0.0476    11
    ## 10   406  -0.0604    12
    ## # ℹ 166 more rows

``` r
cv_df |> pull(train) |> nth(2)
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   390  -0.0504     1
    ##  2   391  -0.0601     2
    ##  3   393  -0.0419     3
    ##  4   396  -0.0599     5
    ##  5   400  -0.0399     8
    ##  6   403  -0.0395    10
    ##  7   405  -0.0476    11
    ##  8   406  -0.0604    12
    ##  9   408  -0.0312    13
    ## 10   409  -0.0382    14
    ## # ℹ 166 more rows

``` r
cv_df |> pull(train) |> nth(3)
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   390  -0.0504     1
    ##  2   391  -0.0601     2
    ##  3   393  -0.0419     3
    ##  4   394  -0.0510     4
    ##  5   397  -0.0284     6
    ##  6   402  -0.0294     9
    ##  7   405  -0.0476    11
    ##  8   406  -0.0604    12
    ##  9   408  -0.0312    13
    ## 10   409  -0.0382    14
    ## # ℹ 166 more rows

Let’s fit models over and over.

``` r
cv_df =
cv_df |> 
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df)),
    smooth_fits = map(train, \(df) mgcv::gam(logratio ~ s(range), data = df)),
    wiggly_fits = map(train, \(df) mgcv::gam(logratio ~ s(range, k = 50), sp = 10e-8, data = df))
  ) |> 
  mutate(
    rmse_linear = map2_dbl(linear_fits, test, rmse),
    rmse_smooth = map2_dbl(smooth_fits, test, rmse),
    rmse_wiggly = map2_dbl(wiggly_fits, test, rmse)
    )
```

Let’s try to look at this better.

``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) +
  geom_violin()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-13-1.png" width="90%" />

## Look at growth data

``` r
growth_df = 
  read_csv("./data/nepalese_children.csv")
```

    ## Rows: 2705 Columns: 5
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (5): age, sex, weight, height, armc
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

Weight v arm_c

``` r
growth_df |>
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = .5)
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-15-1.png" width="90%" />

Let’s show models we might use.

``` r
growth_df =
  growth_df |> 
  mutate(
    weight_cp7 = (weight > 7) * (weight - 7)
  )
```

Let’s fit three models.

``` r
linear_mod = lm(armc ~ weight, data = growth_df)
pwl_mod = lm(armc ~ weight + weight_cp7, data = growth_df)
smooth_mod = mgcv::gam(armc ~ s(weight), data = growth_df)
```

``` r
growth_df |> 
  add_predictions(linear_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = .5) +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-18-1.png" width="90%" />

``` r
growth_df |> 
  add_predictions(pwl_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = .5) +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-19-1.png" width="90%" />

``` r
growth_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = weight, y = armc)) +
  geom_point(alpha = .5) +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-20-1.png" width="90%" />

Now cross validate.

``` r
cv_df =
  crossv_mc(growth_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

``` r
cv_df =
  cv_df |> 
  mutate(
    linear_mod = map(train, \(df) lm(armc ~ weight, data = df)),
    pwl_mod = map(train, \(df) lm(armc ~ weight + weight_cp7, data = df)),
    smooth_mod = map(train, \(df) mgcv::gam(armc ~ s(weight), data = df))
  ) |> 
  mutate(
    rmse_linear = map2_dbl(linear_mod, test, rmse),
    rmse_pwl = map2_dbl(pwl_mod, test, rmse),
    rmse_smooth = map2_dbl(smooth_mod, test, rmse)
  )
```

Create boxplots.

``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse)) +
  geom_violin()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-23-1.png" width="90%" />
