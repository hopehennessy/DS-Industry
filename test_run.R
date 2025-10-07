cat("Running R test script...\n")

# Print basic session info
cat(paste0("R version: ", R.version.string, "\n"))

# Simple computation test
x <- 1:10
m <- mean(x)
cat(paste0("Mean of 1:10 is ", m, "\n"))
stopifnot(abs(m - 5.5) < 1e-12)

# Create a small data frame and write to disk
results <- data.frame(
  value = x,
  squared = x^2
)

out_file <- "test_output.csv"
write.csv(results, out_file, row.names = FALSE)
cat(paste0("Wrote ", out_file, " (", nrow(results), " rows)\n"))

cat("Test completed successfully.\n")
