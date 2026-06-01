#!/usr/bin/env Rscript
# Fit GAMLSS SHASH models per symbol per volatility regime to VECM residuals.
#
# Usage:
#   Rscript gamlss_fit.R --input /path/to/input.csv --output /path/to/output.csv
#
# Setup (run once in R):
#   install.packages("gamlss")
#
# Input CSV columns : date, symbol, residual, regime (Low/Medium/High)
# Output CSV columns: symbol, regime, mu, sigma, nu, tau, q10, q25, q50, q75, q90, n_obs
#
# SHASH (sinh-arcsinh) family is used instead of BCT because VECM residuals are
# centred around zero and can be negative; SHASH supports the full real line with
# the same 4-parameter flexibility (location, scale, skewness, tail-weight).

suppressPackageStartupMessages(library(gamlss))

MIN_OBS <- 10

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

get_arg <- function(args, flag) {
  idx <- which(args == flag)
  if (length(idx) == 0 || idx == length(args)) {
    stop(paste("Missing required argument:", flag))
  }
  args[idx + 1]
}

args        <- commandArgs(trailingOnly = TRUE)
input_path  <- get_arg(args, "--input")
output_path <- get_arg(args, "--output")

# ---------------------------------------------------------------------------
# Load and validate input
# ---------------------------------------------------------------------------

df <- read.csv(input_path, stringsAsFactors = FALSE)

required_cols <- c("date", "symbol", "residual", "regime")
missing_cols  <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
  stop(paste("Input CSV missing columns:", paste(missing_cols, collapse = ", ")))
}

symbols <- unique(df$symbol)
regimes <- c("Low", "Medium", "High")

# ---------------------------------------------------------------------------
# Fit SHASH per symbol × regime
# ---------------------------------------------------------------------------

NA_row <- function(sym, reg, n) {
  data.frame(
    symbol = sym, regime = reg,
    mu = NA_real_, sigma = NA_real_, nu = NA_real_, tau = NA_real_,
    q10 = NA_real_, q25 = NA_real_, q50 = NA_real_,
    q75 = NA_real_, q90 = NA_real_,
    n_obs = n,
    stringsAsFactors = FALSE
  )
}

results <- list()

for (sym in symbols) {
  for (reg in regimes) {
    subset_data <- df[df$symbol == sym & df$regime == reg, ]
    n           <- nrow(subset_data)

    if (n < MIN_OBS) {
      cat(sprintf(
        "WARN  skipping %s / %s: %d obs < %d minimum\n", sym, reg, n, MIN_OBS
      ))
      results[[length(results) + 1]] <- NA_row(sym, reg, n)
      next
    }

    # RS (Rigby-Stasinopoulos) algorithm with n.cyc=100.
    # suppressWarnings: RS may warn "not yet converged" on small samples (~36 obs);
    # the estimate is still near the MLE and preferable to switching optimisers,
    # which risks degenerate solutions on sparse data.
    fit <- tryCatch(
      suppressWarnings(
        gamlss(
          residual ~ 1,
          sigma.formula = ~1,
          nu.formula    = ~1,
          tau.formula   = ~1,
          family        = SHASH(),
          data          = subset_data,
          trace         = FALSE,
          control       = gamlss.control(n.cyc = 100, trace = FALSE)
        )
      ),
      error = function(e) {
        cat(sprintf("WARN  GAMLSS failed for %s / %s: %s\n", sym, reg, conditionMessage(e)))
        NULL
      }
    )

    # Sanity-check: flag degenerate fits (nu escaping to extreme values is a
    # known SHASH instability on very small samples)
    if (!is.null(fit)) {
      nu_val_check <- fitted(fit, "nu")[1]
      if (abs(nu_val_check) > 20) {
        cat(sprintf("WARN  degenerate nu=%.1f for %s / %s — using NA\n",
                    nu_val_check, sym, reg))
        fit <- NULL
      }
    }

    if (is.null(fit)) {
      results[[length(results) + 1]] <- NA_row(sym, reg, n)
      next
    }

    mu_val    <- fitted(fit, "mu")[1]
    sigma_val <- fitted(fit, "sigma")[1]
    nu_val    <- fitted(fit, "nu")[1]
    tau_val   <- fitted(fit, "tau")[1]

    probs  <- c(0.10, 0.25, 0.50, 0.75, 0.90)
    quants <- tryCatch(
      qSHASH(probs, mu = mu_val, sigma = sigma_val, nu = nu_val, tau = tau_val),
      error = function(e) {
        cat(sprintf("WARN  qSHASH failed for %s / %s: %s\n", sym, reg, conditionMessage(e)))
        rep(NA_real_, 5)
      }
    )

    results[[length(results) + 1]] <- data.frame(
      symbol = sym, regime = reg,
      mu     = mu_val,    sigma = sigma_val,
      nu     = nu_val,    tau   = tau_val,
      q10    = quants[1], q25   = quants[2],
      q50    = quants[3], q75   = quants[4],
      q90    = quants[5],
      n_obs  = n,
      stringsAsFactors = FALSE
    )

    cat(sprintf(
      "OK    %s / %-6s  n=%d  mu=%.4f  sigma=%.4f  nu=%.4f  tau=%.4f\n",
      sym, reg, n, mu_val, sigma_val, nu_val, tau_val
    ))
  }
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

output_df <- do.call(rbind, results)
write.csv(output_df, output_path, row.names = FALSE)
cat(sprintf("INFO  wrote %d rows to %s\n", nrow(output_df), output_path))
