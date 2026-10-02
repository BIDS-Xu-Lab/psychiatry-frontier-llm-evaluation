# =============================================================================
# Figures for the NEJM AI resubmission — R / ggplot2
#
# Figure 2  Diagnostic accuracy and reasoning quality dissociate.
#           Paired within-case differences, six model pairs, Holm-corrected.
# Figure 3  Computable trace features do not substitute for clinician rating.
#           (A) model-level association, n = 4
#           (B) explanatory ceiling of each feature's within-model slope
#               against the observed reasoning gaps
#
# All numbers are the VERIFIED values from COMPLETE_ANALYSIS_DOCUMENT_v2.md
# (§4 pairwise table, §5 TOST bound, §9 proxy tables). Replace the inline
# tibbles with read.csv() calls to regenerate from source.
#
# Requires: ggplot2 (>= 3.4), patchwork.
# Output:   fig2_dissociation.{png,pdf}, fig3_proxies.{png,pdf}  (300 dpi)
# =============================================================================

library(ggplot2)
library(patchwork)

# ---- House style -------------------------------------------------------------
INK    <- "#1a1a1a"
ACCENT <- "#b3392b"   # only the pair carrying the primary claim
GREY   <- "#8a8a8a"
BAND   <- "#e6e6e6"

theme_ms <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      panel.grid       = element_blank(),
      axis.line        = element_line(colour = INK, linewidth = 0.4),
      axis.ticks       = element_line(colour = INK, linewidth = 0.4),
      axis.text        = element_text(colour = INK),
      plot.title       = element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.title.position = "plot",
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.text      = element_text(size = base_size - 1),
      plot.margin      = margin(6, 8, 6, 6)
    )
}

# =============================================================================
# FIGURE 2 — verified pairwise differences (§4) and TOST bound (§5)
# =============================================================================

pairs <- data.frame(
  pair    = c("Claude vs GPT-5.2", "Gemini vs GPT-5.2", "Claude vs DeepSeek",
              "DeepSeek vs GPT-5.2", "Claude vs Gemini", "DeepSeek vs Gemini"),
  d_acc   = c(+0.060, +0.053, +0.000, +0.060, +0.007, +0.007),
  acc_lo  = c(-0.079, -0.052, -0.071, -0.090, -0.090, -0.117),
  acc_hi  = c(+0.199, +0.159, +0.071, +0.210, +0.104, +0.130),
  d_rsn   = c(+1.23,  +0.75,  +0.70,  +0.53,  +0.48,  -0.22),
  rsn_lo  = c(+1.04,  +0.52,  +0.50,  +0.27,  +0.29,  -0.52),
  rsn_hi  = c(+1.43,  +0.99,  +0.90,  +0.80,  +0.67,  +0.08),
  holm_p  = c(0.001,  0.001,  0.001,  0.0006, 0.0001, 0.138),
  stringsAsFactors = FALSE
)
TOST_BOUND      <- 0.059   # smallest margin at which Claude/DeepSeek passes, unadjusted
TOST_BOUND_HOLM <- 0.10    # after Holm adjustment
PRIMARY_PAIR    <- "Claude vs DeepSeek"

# Keep listed order top-to-bottom; use numeric y so annotate() rectangles and
# points share one continuous scale
pairs$y       <- rev(seq_len(nrow(pairs)))
pairs$primary <- pairs$pair == PRIMARY_PAIR
pairs$rsn_sig <- pairs$holm_p < 0.05
pairs$colour  <- ifelse(pairs$primary, "primary", "other")
# Accuracy: no pair is separated, so every point is open. Reasoning: open iff n.s.
pairs$acc_fill <- "not separated"
pairs$rsn_fill <- ifelse(pairs$rsn_sig, "separated (Holm p < 0.05)", "not separated")

colour_scale <- scale_colour_manual(
  values = c(primary = ACCENT, other = INK),
  breaks = "primary", labels = "pair carrying primary claim"
)
fill_scale <- scale_fill_manual(
  values = c("separated (Holm p < 0.05)" = INK, "not separated" = "white"),
  breaks = c("separated (Holm p < 0.05)", "not separated")
)

band_y <- pairs$y[pairs$primary]   # y position of the primary row

pA <- ggplot(pairs, aes(y = y)) +
  # equivalence bounds behind the primary row only
  annotate("rect", xmin = -TOST_BOUND_HOLM, xmax = TOST_BOUND_HOLM,
           ymin = band_y - 0.38, ymax = band_y + 0.38, fill = BAND) +
  annotate("rect", xmin = -TOST_BOUND, xmax = TOST_BOUND,
           ymin = band_y - 0.38, ymax = band_y + 0.38, fill = "#d0d0d0") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = GREY, linewidth = 0.4) +
  geom_errorbarh(aes(xmin = acc_lo, xmax = acc_hi, colour = colour),
                 height = 0, linewidth = 0.6) +
  geom_point(aes(x = d_acc, colour = colour, fill = acc_fill),
             shape = 21, size = 2.4, stroke = 0.7) +
  annotate("text", x = -0.245, y = band_y,
           label = "equivalence\nbounds \u00B10.06\n(\u00B10.10 Holm)",
           hjust = 0, vjust = 0.5, size = 2.2, colour = GREY, lineheight = 0.85) +
  colour_scale + fill_scale +
  scale_x_continuous(limits = c(-0.25, 0.25), breaks = seq(-0.2, 0.2, 0.1)) +
  scale_y_continuous(breaks = pairs$y, labels = pairs$pair, expand = expansion(add = 0.7)) +
  labs(title = "A  Accuracy: 0 of 6 pairs separated",
       x = "Difference in clinician-adjudicated accuracy\n(proportion of cases, 95% CI)",
       y = NULL) +
  theme_ms() +
  guides(colour = "none", fill = "none")

pB <- ggplot(pairs, aes(y = y)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = GREY, linewidth = 0.4) +
  geom_errorbarh(aes(xmin = rsn_lo, xmax = rsn_hi, colour = colour),
                 height = 0, linewidth = 0.6) +
  geom_point(aes(x = d_rsn, colour = colour, fill = rsn_fill),
             shape = 21, size = 2.4, stroke = 0.7) +
  colour_scale + fill_scale +
  scale_x_continuous(limits = c(-0.75, 1.6), breaks = seq(-0.5, 1.5, 0.5)) +
  scale_y_continuous(breaks = pairs$y, labels = NULL, expand = expansion(add = 0.7)) +
  labs(title = "B  Reasoning: 5 of 6 pairs separated",
       x = "Difference in clinician-rated reasoning quality\n(0\u20134 scale, 95% CI)",
       y = NULL) +
  theme_ms() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  guides(
    fill   = guide_legend(order = 1, override.aes = list(colour = INK)),
    colour = guide_legend(order = 2, override.aes = list(shape = NA, linewidth = 1))
  )

fig2 <- (pA | pB) +
  plot_layout(widths = c(1, 1), guides = "collect") &
  theme(legend.position = "bottom", legend.box = "horizontal")

ggsave("fig2_dissociation.png", fig2, width = 7.2, height = 3.6, dpi = 300, bg = "white")
ggsave("fig2_dissociation.pdf", fig2, width = 7.2, height = 3.6, device = cairo_pdf)

# =============================================================================
# FIGURE 3 — proxy analysis (§9)
# =============================================================================

models <- data.frame(
  model    = c("Claude Opus 4.5", "Gemini 3 Pro", "DeepSeek-V3.2", "GPT-5.2"),
  rating   = c(3.60, 3.12, 2.90, 2.37),
  words    = c(1087.8, 431.0, 619.6, 900.2),
  codes    = c(2.83, 3.97, 3.50, 8.77),
  selfcorr = c(0.73, 1.40, 3.77, 3.07),
  stringsAsFactors = FALSE
)
# manual label offsets so DeepSeek doesn't collide with Gemini
models$lab_dx <- 0.25
models$lab_dy <- ifelse(models$model == "DeepSeek-V3.2", -0.12, 0.04)

ceiling <- data.frame(
  feature   = c("Trace length (per 100 words)", "Codes per 1,000 words",
                "Self-corrections per trace", "ICD-10 codes per trace",
                "Self-corrections per 1k words"),
  predicted = c(+0.267, -0.141, +0.127, +0.115, -0.046),
  stringsAsFactors = FALSE
)
ceiling$y <- rev(seq_len(nrow(ceiling)))
OBS_GAP <- c(0.48, 1.23)   # range of separated pairwise reasoning differences

pC <- ggplot(models, aes(x = codes, y = rating)) +
  geom_point(size = 2.6, colour = INK) +
  geom_text(aes(x = codes + lab_dx, y = rating + lab_dy, label = model),
            hjust = 0, size = 2.6, colour = INK) +
  annotate("text", x = 10.8, y = 3.85,
           label = "descriptive only \u2014\nno p < 0.083 attainable at n = 4",
           hjust = 1, vjust = 1, size = 2.4, colour = GREY, lineheight = 0.9) +
  scale_x_continuous(limits = c(1.5, 11), breaks = seq(2, 10, 2)) +
  scale_y_continuous(limits = c(2.1, 3.9), breaks = seq(2.2, 3.8, 0.2)) +
  labs(title = "A  Between models (n = 4): \u03C1 = \u22120.80",
       x = "Mean distinct ICD-10 codes named per trace",
       y = "Mean clinician-rated reasoning (0\u20134)") +
  theme_ms()

pD <- ggplot(ceiling, aes(y = y)) +
  annotate("rect", xmin = OBS_GAP[1], xmax = OBS_GAP[2], ymin = -Inf, ymax = Inf, fill = BAND) +
  annotate("text", x = mean(OBS_GAP), y = 5.55,
           label = "observed reasoning gaps\nbetween separated pairs",
           size = 2.4, colour = GREY, vjust = 0, lineheight = 0.9) +
  geom_vline(xintercept = 0, colour = GREY, linewidth = 0.4) +
  geom_rect(aes(xmin = pmin(0, predicted), xmax = pmax(0, predicted),
                ymin = y - 0.275, ymax = y + 0.275), fill = INK) +
  scale_x_continuous(limits = c(-0.3, 1.35), breaks = seq(-0.25, 1.25, 0.25)) +
  scale_y_continuous(breaks = ceiling$y, labels = ceiling$feature) +
  coord_cartesian(ylim = c(0.4, 6.1), clip = "off") +
  labs(title = "B  Within model (n = 120): explanatory ceiling",
       x = "Predicted reasoning difference, 0\u20134 scale\n(within-model slope \u00D7 largest between-model spread)",
       y = NULL) +
  theme_ms()

fig3 <- (pC | pD) + plot_layout(widths = c(1, 1.15))

ggsave("fig3_proxies.png", fig3, width = 8.4, height = 3.3, dpi = 300, bg = "white")
ggsave("fig3_proxies.pdf", fig3, width = 8.4, height = 3.3, device = cairo_pdf)

# -----------------------------------------------------------------------------
# For a true within-model panel (per-model scatter with the mixed-model slope
# from lme4::lmer(dx ~ icd_codes + model + (1 | case_id))), load the trace-level
# table with columns model, case_id, dx, icd_codes, words, selfcorr and replace
# pD. Not implemented here because trace-level data are not distributed with
# the manuscript.
# -----------------------------------------------------------------------------

cat("wrote fig2_dissociation.{png,pdf} and fig3_proxies.{png,pdf}\n")
