# Analysis of clinician-rated LLM diagnostic reasoning quality (n=30, m=5)
library(irr)
library(lme4)
library(ggeffects)
library(ggplot2)
library(dplyr)

## Calculate interrater agreement
# Extract diagnostic match values
path <- "~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/evaluate_diagnostic_reasoning/clinician_annotations/diagnostic_match_pivot.csv"
diagnostic_match <- read.csv(path)
diagnostic_match <- diagnostic_match[, 3:7]

# Fleiss's kappa for 5 clinician annotators on model diagnosis = true diagnosis
kappam.fleiss(diagnostic_match, detail = TRUE)

# Average percent agreement to supplement
all_agree <- apply(diagnostic_match, 1, function(row) length(unique(row)) == 1)
percent_all_agree <- mean(all_agree)
percent_all_agree

# Extract reasoning scores (extraction and diagnosis scores)
path <- "~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/evaluate_diagnostic_reasoning/clinician_annotations/diagnostic_reasoning_pivot.csv"
reasoning_scores <- read.csv(path)
extraction_scores <- data.frame(lapply(reasoning_scores[-1, 3:7], as.numeric))
diagnosis_scores <- data.frame(lapply(reasoning_scores[-1, 8:12], as.numeric))

# Intraclass correlation coefficient (ICC) for 5 clinician raters on the extraction score
icc(extraction_scores,
    model = "twoway",
    type = "agreement",
    unit = "average")

# ICC for 5 clinician annotators on the diagnosis score
icc(diagnosis_scores,
    model = "twoway",
    type = "agreement",
    unit = "average")

## Measure correlation between diagnostic match and reasoning quality
path <- "~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/evaluate_diagnostic_reasoning/clinician_annotations/processed_full_list.csv"
full_list <- read.csv(path)

# Convert variables to proper coded types
full_list$diagnosis_match <- ifelse(full_list$diagnosis_match == "Yes", 1, 0)

# Fit a generalized linear mixed-effects model
model_mixed <- glmer(
  diagnosis_match ~ reasoning_diagnosis_score + reasoning_extraction_score + model_name + (1 | annotator) + (1 | case_id),
  data = full_list,
  family = binomial
)
summary(model_mixed)

# Odds ratio
exp(fixef(model_mixed)[2])
# Interpretation: Each one-point increase in clinician-rated diagnostic reasoning
# quality was associated with a 6.06x increase in the odds of a correct diagnosis.

# Visualization
# Generate predictions from the regression model for diagnosis score
pred_reasoning <- ggpredict(
  model_mixed,
  terms = "reasoning_diagnosis_score [0:4]",
  bias_correction = TRUE
)

# Plot the predicted diagnosis match probability
ggplot(pred_reasoning, aes(x, predicted)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(
    x = "Clinician-rated diagnostic reasoning score",
    y = "Predicted probability of correct diagnosis"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal()

# Generate predictions from the regression model for extraction score
pred_extraction <- ggpredict(
  model_mixed,
  terms = "reasoning_extraction_score [0:4]"
)

# Plot the predicted diagnosis match probability
ggplot(pred_extraction, aes(x, predicted)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(
    x = "Clinician-rated data extraction score",
    y = "Predicted probability of correct diagnosis"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal()

# Show that reasoning-correctness holds across LLMs
pred_by_model <- ggpredict(
  model_mixed,
  terms = c("reasoning_diagnosis_score [0:4]", "model_name")
)

ggplot(pred_by_model, aes(x, predicted, color = group)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.15) +
  labs(
    x = "Clinician-rated diagnostic reasoning score",
    y = "Predicted probability of correct diagnosis",
    color = "Model",
    fill = "Model",
    title = "Clinician-rated diagnostic reasoning quality correlates with probability of a\ncorrectly diagnosed psychiatric case vignette in large language models\n(n = 30)"
  ) +
  theme_minimal() + 
  theme(axis.text = element_text(size = 12),
        plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 10),
        legend.position=c(.85,.25))

plot_path <- "~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/analysis"
ggsave("figure_2.png",  path = plot_path)

# Boxplots by correctness
ggplot(full_list, aes(factor(diagnosis_match), reasoning_diagnosis_score)) +
  geom_boxplot() +
  labs(
    x = "Diagnostic correctness",
    y = "Clinician-rated diagnostic reasoning score"
  ) +
  scale_x_discrete(labels = c("No", "Yes")) +
  theme_minimal()

## Calculate average reasoning scores
case_model_means <- full_list %>%
  group_by(case_id, model_name) %>%
  summarise(
    mean_reasoning = mean(reasoning_diagnosis_score, na.rm = TRUE),
    mean_extraction = mean(reasoning_extraction_score, na.rm = TRUE),
    .groups = "drop"
  )

model_means <- case_model_means %>%
  group_by(model_name) %>%
  summarise(
    avg_reasoning  = mean(mean_reasoning),
    sd_reasoning   = sd(mean_reasoning),
    avg_extraction = mean(mean_extraction),
    sd_extraction  = sd(mean_extraction),
    n_cases        = n(),
    .groups = "drop"
  )

model_means
