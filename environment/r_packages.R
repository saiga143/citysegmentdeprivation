# Install helper for the VSURF feature selection stage.
#
# Run this script once in an R session before executing run_VSURF.R:
#   source("environment/r_packages.R")
#
# Packages:
#   data.table  - fast CSV loading for the labeled training files
#   VSURF       - Variable Selection Using Random Forests
#   VennDiagram - Venn diagram of variable selections across ntree settings
#   gridExtra   - multi-panel plot layout

required_packages <- c("data.table", "VSURF", "VennDiagram", "gridExtra")

install_if_missing <- function(pkgs, repos = "https://cloud.r-project.org") {
  missing_pkgs <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing_pkgs) == 0) {
    message("All required packages are already installed.")
  } else {
    message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
    install.packages(missing_pkgs, repos = repos)
  }
  invisible(NULL)
}

install_if_missing(required_packages)
