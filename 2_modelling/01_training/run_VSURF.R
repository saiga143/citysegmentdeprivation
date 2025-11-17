# ============================================================
# VSURF Feature Selection on Labeled Training Data (8 Cities)
# ------------------------------------------------------------
# This script:
#   1. Loads all *_labeled_thr030.csv files produced in the 
#      "Create Labeled Data" preprocessing step.
#   2. Runs VSURF with ntree = 800, 1000, 1200.
#   3. Prints selected prediction and interpretation variables.
#   4. Displays a Venn diagram comparing varselect.pred sets.
#
# NOTE:
#   - No outputs are saved to disk. 
#   - All results are displayed directly in the R session.
#   - VSURF is R-only, so this script is uploaded as an R file.
#
# INPUT LOCATION (relative):
#     ../1_preprocessing/LabelledData_For_RF/
#
# These CSVs *are included* in the GitHub repo.
# ============================================================

# --- Safeguard: prevent re-running in same R session ----------
if (exists("vsurf_script_has_run") && isTRUE(vsurf_script_has_run)) {
  stop("🚫 This script has already run in this R session.")
}
vsurf_script_has_run <- TRUE

# --- 1. Load required libraries --------------------------------
suppressPackageStartupMessages({
  library(data.table)
  library(VSURF)
  library(VennDiagram)
  library(gridExtra)
})

# --- 2. Set folder path (relative) ------------------------------
folder_path <- "../1_preprocessing/LabelledData_For_RF"

message("📁 Using labeled training data from: ", normalizePath(folder_path))

# --- 3. Read all labeled CSVs ----------------------------------
csv_files <- list.files(
  folder_path,
  pattern = "_labeled_thr030\\.csv$",
  full.names = TRUE
)

if (length(csv_files) == 0) {
  stop("No *_labeled_thr030.csv files found in: ", folder_path)
}

message("Found ", length(csv_files), " labeled training files.")

data_list <- lapply(csv_files, fread)
data <- rbindlist(data_list, fill = TRUE)

# --- 4. Keep only rows with labels ------------------------------
data <- data[!is.na(slum_label1)]
data$slum_label1 <- as.factor(data$slum_label1)

# --- 5. Feature list --------------------------------------------
features <- c(
  "POP_SEG", "AREAHA_SEG", "ROAD_SEG",
  "PAR_N_SEG", "PARU_N_SEG", "PARU_P_SEG",
  "PARU_A_SEG", "PAR_CV_SEG",
  "B_AREA_SEG", "B_AVG_SEG", "B_CV_SEG",
  "i1_pop_area", "i2_pop_par", "i3_pop_paru", "i4_pop_roads",
  "i5_par_area", "i6_paru_area", "i7_roads_area",
  "i8_paru_par", "i9_roads_par", "i10_roads_paru",
  "REG1_GHSL"
)

missing_feats <- setdiff(features, colnames(data))
if (length(missing_feats) > 0) {
  stop("Missing feature columns in labeled data: ",
       paste(missing_feats, collapse = ", "))
}

# Extract predictors and target
X <- data[, ..features]
y <- data$slum_label1

# Remove NA rows
complete <- complete.cases(X)
X <- X[complete, ]
y <- y[complete]

message("Rows after NA filtering: ", nrow(X))

# --- 6. Function to run VSURF -----------------------------------
run_vsurf <- function(ntree_value) {
  set.seed(42)
  result <- VSURF(
    X, y,
    ntree    = ntree_value,
    parallel = TRUE
  )
  
  list(
    pred   = colnames(X)[result$varselect.pred],
    interp = colnames(X)[result$varselect.interp]
  )
}

# --- 7. Run VSURF for multiple ntree values ----------------------
ntree_values <- c(800, 1000, 1200)
results <- list()

for (nt in ntree_values) {
  cat("\n🔄 Running VSURF with ntree =", nt, "...\n")
  res <- run_vsurf(nt)
  results[[as.character(nt)]] <- res
  cat("✅ Completed ntree =", nt, 
      "| Selected predictors:", length(res$pred), "\n")
}

# --- 8. Display selected variables -------------------------------
cat("\n📊 VSURF Selected Variables Summary:\n")

for (nt in ntree_values) {
  key <- as.character(nt)
  cat("\n🟦 ntree =", nt, "\n")
  cat("Prediction variables:\n")
  print(results[[key]]$pred)
  cat("\nInterpretation variables:\n")
  print(results[[key]]$interp)
}

# --- 9. Venn Diagram of varselect.pred ---------------------------
venn_sets <- list()
for (nt in ntree_values) {
  venn_sets[[paste0("ntree_", nt)]] <- results[[as.character(nt)]]$pred
}

if (length(venn_sets) == 3) {
  grid.newpage()
  draw.triple.venn(
    area1 = length(venn_sets[[1]]),
    area2 = length(venn_sets[[2]]),
    area3 = length(venn_sets[[3]]),
    n12   = length(intersect(venn_sets[[1]], venn_sets[[2]])),
    n23   = length(intersect(venn_sets[[2]], venn_sets[[3]])),
    n13   = length(intersect(venn_sets[[1]], venn_sets[[3]])),
    n123  = length(Reduce(intersect, venn_sets)),
    category = names(venn_sets),
    fill     = c("skyblue", "salmon", "palegreen"),
    alpha    = 0.5,
    cex      = 1.4,
    cat.cex  = 1.2
  )
} else {
  cat("⚠️ Venn diagram shown only for 3 ntree sets.\n")
}

cat("\n🎉 VSURF feature selection complete.\n")
