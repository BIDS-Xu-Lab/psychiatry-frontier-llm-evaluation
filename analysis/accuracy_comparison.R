library(dplyr)
library(lme4)
library(ggplot2)
library(binom)

# Set working directory
setwd("~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/top_5_accuracy/accuracy_metrics/memorization_experiment/detailed_results/")

# Filemap
files_by_model <- list(
  "DeepSeek-Reasoner" = list(
    fictitious = "predicted_diagnoses_deepseek-reasoner_fictitious_only_20260126_142132.json_diagnostic_evaluation_results_detailed.csv",
    literature = "predicted_diagnoses_deepseek-reasoner_medical_literature_only_20260126_145408.json_diagnostic_evaluation_results_detailed.csv"
  ),
  "Claude-Opus-4.5" = list(
    fictitious = "predicted_diagnoses_claude-opus-4-5-20251101_fictitious_only_20260126_201438.json_diagnostic_evaluation_results_detailed.csv",
    literature = "predicted_diagnoses_claude-opus-4-5-20251101_medical_literature_only_20260126_212746.json_diagnostic_evaluation_results_detailed.csv"
  ),
  "GPT-5.2" = list(
    fictitious = "predicted_diagnoses_gpt-5.2_fictitious_only_20260126_180947.json_diagnostic_evaluation_results_detailed.csv",
    literature = "predicted_diagnoses_gpt-5.2_medical_literature_only_20260127_013541.json_diagnostic_evaluation_results_detailed.csv"
  ),
  "Gemini-3-Pro" = list(
    fictitious = "predicted_diagnoses_gemini-3-pro-preview_fictitious_only_20260126_151138.json_diagnostic_evaluation_results_detailed.csv",
    literature = "predicted_diagnoses_gemini-3-pro-preview_medical_literature_only_20260126_163757.json_diagnostic_evaluation_results_detailed.csv"
  )
)

# -----------------------------
# Helpers
# -----------------------------
read_one <- function(path, model_name, vignette_source_label) {
  read.csv(path, stringsAsFactors = FALSE) %>%
    mutate(
      model_name = model_name,
      vignette_source = vignette_source_label
    )
}

extract_or_ci <- function(m, term = "vignette_sourcemedical_literature") {
  coefs <- summary(m)$coefficients
  if (!(term %in% rownames(coefs))) {
    return(data.frame(beta = NA, OR = NA, CI_low = NA, CI_high = NA, p_value = NA))
  }
  beta <- coefs[term, "Estimate"]
  pval <- coefs[term, "Pr(>|z|)"]
  OR <- exp(beta)
  CI <- exp(confint(m, parm = term, method = "Wald"))
  data.frame(beta = beta, OR = OR, CI_low = CI[1], CI_high = CI[2], p_value = pval)
}

# -----------------------------
# 1) Build a unified case-level dataset
# -----------------------------
all_rows <- list()

for (mn in names(files_by_model)) {
  cat("Processing:", mn, "\n")
  
  f_fict <- files_by_model[[mn]]$fictitious
  f_lit  <- files_by_model[[mn]]$literature
  
  df_fict <- read_one(f_fict, mn, "fictitious")
  df_lit  <- read_one(f_lit,  mn, "medical_literature")
  
  df <- bind_rows(df_fict, df_lit) %>%
    transmute(
      case_id = as.character(case_id),
      model_name = mn,
      vignette_source = vignette_source,
      correct_top1 = as.integer(round(as.numeric(hybrid_top1)))
    ) %>%
    filter(!is.na(case_id), !is.na(vignette_source), !is.na(correct_top1)) %>%
    mutate(
      case_id = factor(case_id),
      model_name = factor(model_name, levels = names(files_by_model)),
      vignette_source = factor(vignette_source, levels = c("fictitious", "medical_literature"))
    )
  
  # Sanity checks
  cat("  rows:", nrow(df),
      "| sources:", paste(names(table(df$vignette_source)), table(df$vignette_source), collapse = ", "),
      "| unique cases:", dplyr::n_distinct(df$case_id),
      "\n")
  
  all_rows[[mn]] <- df
}

df_all <- bind_rows(all_rows)

# -----------------------------
# 2) Per-model mixed-effects logistic regression
# -----------------------------
reg_rows <- list()

for (mn in levels(df_all$model_name)) {
  dat <- df_all %>% filter(model_name == mn)
  
  # Fit: correct_top1 ~ vignette_source + (1|case_id)
  m <- glmer(
    correct_top1 ~ vignette_source + (1 | case_id),
    data = dat,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa")
  )
  
  r <- extract_or_ci(m, term = "vignette_sourcemedical_literature")
  reg_rows[[mn]] <- data.frame(
    model_name = mn,
    OR_literature_vs_fictitious = round(r$OR, 3),
    CI_low = round(r$CI_low, 3),
    CI_high = round(r$CI_high, 3),
    p_value = signif(r$p_value, 3)
  )
}

reg_table <- bind_rows(reg_rows)
print(reg_table)

write.csv(reg_table, "supplement_source_effect_top1_by_model.csv", row.names = FALSE)

# -----------------------------
# 3) Optional pooled interaction test
# -----------------------------
m_noint <- glmer(
  correct_top1 ~ vignette_source + model_name + (1 | case_id),
  data = df_all,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

m_int <- glmer(
  correct_top1 ~ vignette_source * model_name + (1 | case_id),
  data = df_all,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

lrt <- anova(m_noint, m_int, test = "LRT")
print(lrt)

lrt_out <- data.frame(
  comparison = "no_interaction vs interaction",
  Chisq = lrt$Chisq[2],
  df = lrt$`Chi Df`[2],
  p_value = lrt$`Pr(>Chisq)`[2]
)
write.csv(lrt_out, "supplement_source_effect_top1_interaction_LRT.csv", row.names = FALSE)

# -----------------------------
# 4) Accuracy table + 95% CI plot (Wilson)
# -----------------------------
acc <- df_all %>%
  group_by(model_name, vignette_source) %>%
  summarise(
    correct = sum(correct_top1 == 1),
    n = n(),
    acc = correct / n,
    .groups = "drop"
  )

ci <- binom.confint(acc$correct, acc$n, methods = "wilson")
acc$ci_low <- ci$lower
acc$ci_high <- ci$upper

acc_out <- acc %>%
  mutate(
    acc = round(acc, 3),
    ci_low = round(ci_low, 3),
    ci_high = round(ci_high, 3)
  )

write.csv(acc_out, "supplement_top1_accuracy_by_source_all_models_table.csv", row.names = FALSE)

p <- ggplot(acc, aes(x = vignette_source, y = acc)) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15) +
  facet_wrap(~ model_name) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = "Vignette source",
    y = "Top-1 accuracy (95% CI)",
    title = "Top-1 diagnostic accuracy by vignette source"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5),
    strip.background = element_rect(fill = "white")
  )

print(p)

ggsave(
  "supplement_top1_accuracy_by_source_all_models.png",
  p,
  width = 8.5,
  height = 5.5,
  dpi = 300
)

message("Wrote:\n",
        "- supplement_source_effect_top1_by_model.csv\n",
        "- supplement_source_effect_top1_interaction_LRT.csv\n",
        "- supplement_top1_accuracy_by_source_all_models_table.csv\n",
        "- supplement_top1_accuracy_by_source_all_models.png\n")