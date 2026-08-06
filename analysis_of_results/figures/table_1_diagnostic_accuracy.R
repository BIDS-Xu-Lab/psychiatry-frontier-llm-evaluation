library(gt)
library(gtsummary)
library(tidyverse)
library(webshot2)

# Read in CSV files
setwd("~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/results/top_5_accuracy/accuracy_metrics/summarized_results")
gpt_5.2 <- read.csv("./predicted_diagnoses_gpt-5.2_20251218_122902.json_diagnostic_performance_summary.csv")
gemini_3_pro_preview <- read.csv("./predicted_diagnoses_gemini-3-pro-preview_20251217_184205.json_diagnostic_performance_summary.csv")
deepseek_reasoner <- read.csv("./predicted_diagnoses_deepseek-reasoner_20251215_215332.json_diagnostic_performance_summary.csv")
claude_opus_4.5 <- read.csv("./predicted_diagnoses_claude-opus-4-5-20251101_20251215_225418.json_diagnostic_performance_summary.csv")

# Combine data frames into a tibble
combined_results <- rbind(
  data.frame(Model = "OpenAI GPT-5.2", gpt_5.2),
  data.frame(Model = "Google Gemini 3 Pro", gemini_3_pro_preview),
  data.frame(Model = "DeepSeek-V3.2", deepseek_reasoner),
  data.frame(Model = "Anthropic Claude Opus 4.5", claude_opus_4.5)
) |> as_tibble()


#### Long format table ####
# Sort data by score
combined_results <-
  combined_results |>
  group_by(Model) |>
  arrange(desc(Score), .by_group = TRUE)

# Convert to gt table
gt_results <- gt(combined_results,
                 rowname_col = "Metric")

# Add header
gt_results <-
  gt_results |>
  tab_header(
    title = "Psychiatric Diagnostic Accuracy of Large Language Models",
    subtitle = md("Based on ability to diagnose psychiatric case vignettes (*n* = 196)")
  )

# Round scores to 3 decimal places
gt_results <- 
  gt_results |>
  fmt_number(columns = Score,
             decimals = 3) |>
  cols_label(Score = md("Mean Score"))

# Add row groups
gt_results <-
  gt_results |>
  tab_stubhead(label = md("Models")) |>
  tab_stub_indent(rows = everything(),
                  indent = 5)

# Add footnotes
gt_results <-
  gt_results |>
  tab_footnote(
    footnote = md("**Models accessed via respective provider APIs in December 2025.**"),
    locations = cells_title(groups = "title")
  ) |>
  tab_footnote(
    footnote = md("**Includes literature-derived and fictitious case vignettes.**"),
    locations = cells_title(groups = "subtitle")
  ) |>
  # tab_source_note(
  #   source_note = md("**Diagnostic performance evaluated via real-world and fictitious case vignettes.**")
  # ) |>
  tab_footnote(
    footnote = "Did the model identify the correct primary diagnosis?",
    locations = cells_stub(rows = c("Top-1 Accuracy"))
  ) |>
  tab_footnote(
    footnote = "Did the model identify at least one correct diagnosis among its 5 predictions?",
    locations = cells_stub(rows = c("Top-5 Accuracy"))
  ) |>
  tab_footnote(
    footnote = "Percentage of patient's total diagnoses identified by the model.",
    locations = cells_stub(rows = c("Recall@5"))
  ) |>
  tab_footnote(
    footnote = "How high up did the model's first correct diagnosis appear?",
    locations = cells_stub(rows = c("Mean Reciprocal Rank"))
  ) |>
  opt_footnote_marks(marks = "standard")

# Style cells based on value
gt_results <-
  gt_results |>
  # Bold the model names
  tab_style(
    style = "font-variant: small-caps;",
    locations = cells_row_groups(groups = everything())) |>
  # Highlight top-scoring model
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      columns = Score,
      rows = 1:4
    )
  ) |>
  # Style the column names
  tab_style(
    style = cell_text(weight = "bold"),
    locations = list(cells_stubhead(), cells_column_labels())
  )
gt_results

# Export as PNG
setwd("~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/analysis")
gtsave(gt_results, filename = "table_1_long.png")



#### Wide format table ####
# Pivot models values into their own columns
combined_results_wide <-
  combined_results |>
  pivot_wider(names_from = Model, values_from = Score) |>
  arrange(Metric)

gt_results_wide <- gt(combined_results_wide)

# Add header
gt_results_wide <-
  gt_results_wide |>
  tab_header(
    title = "Psychiatric Diagnostic Accuracy of Large Language Models",
    subtitle = md("Based on ability to diagnose psychiatric case vignettes (*n* = 196)")
  )

# Round scores to 3 decimal places
gt_results_wide <- 
  gt_results_wide |>
  fmt_number(columns = everything(),
             decimals = 3)

# Order columns by score
gt_results_wide <-
  gt_results_wide |>
  cols_move(
    columns = "Google Gemini 3 Pro",
    after = "OpenAI GPT-5.2"
  )

# Relabel columns
gt_results_wide <-
  gt_results_wide |>
  cols_label(
    c("Anthropic Claude Opus 4.5") ~ md("Claude Opus 4.5"),
    c("Google Gemini 3 Pro") ~ md("Gemini 3 Pro"),
    c("OpenAI GPT-5.2") ~ md("GPT-5.2"),
    c("Metric") ~ ""  # Remove "Metric" label
  )

# Add footnotes
gt_results_wide <-
  gt_results_wide |>
  tab_footnote(
    footnote = md("Models accessed via respective provider APIs in December 2025."),
    locations = cells_title(groups = "title")
  ) |>
  tab_footnote(
    footnote = md("Rounded to three decimal places."),
    locations = cells_title(groups = "subtitle")
  ) |>
  tab_footnote(
    footnote = "Did the model identify the correct primary diagnosis?",
    locations = cells_body(
      columns = "Metric",
      rows = 3
      )
  ) |>
  tab_footnote(
    footnote = "Did the model identify at least one correct diagnosis among its 5 predictions?",
    locations = cells_body(
      columns = "Metric",
      rows = 4
    )
  ) |>
  tab_footnote(
    footnote = "Percentage of patient's total diagnoses identified by the model.",
    locations = cells_body(
      columns = "Metric",
      rows = 2
    )
  ) |>
  tab_footnote(
    footnote = "How high up did the model's first correct diagnosis appear?",
    locations = cells_body(
      columns = "Metric",
      rows = 1
    )
  ) |>
  opt_footnote_marks(marks = "standard") |>
  tab_options(
    footnotes.font.size = "8pt" # Or use specific points like "8pt"
  )

# Style cells based on value
gt_results_wide <-
  gt_results_wide |>
  # Highlight top-scoring model
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(
      columns = "Anthropic Claude Opus 4.5"
      )
    ) |>
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(
      columns = "Anthropic Claude Opus 4.5"
    ) 
  ) 

gt_results_wide
setwd("~/Documents/School/PhD (Yale)/PhD/xu_lab/Repositories/psychiatry-frontier-llm-evaluation/analysis")
gtsave(gt_results_wide, filename = "table_1_wide.png")
