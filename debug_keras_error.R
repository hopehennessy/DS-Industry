# Debug Keras Error
library(keras)
library(tensorflow)

# Create simple test data
n_users <- 10
n_items <- 5
n_samples <- 20

# Create test data
set.seed(123)
user_indices <- sample(0:(n_users-1), n_samples, replace = TRUE)
item_indices <- sample(0:(n_items-1), n_samples, replace = TRUE)
ratings <- runif(n_samples, 1, 10)

cat("Test data created:\n")
cat("User indices:", class(user_indices), "length:", length(user_indices), "\n")
cat("Item indices:", class(item_indices), "length:", length(item_indices), "\n")
cat("Ratings:", class(ratings), "length:", length(ratings), "\n")

# Test different input formats
cat("\n=== Testing Input Format 1: Named List ===\n")
tryCatch({
  x_test1 <- list(
    user_input = array(as.integer(user_indices), dim = c(length(user_indices), 1)),
    item_input = array(as.integer(item_indices), dim = c(length(item_indices), 1))
  )
  cat("Format 1 created successfully\n")
  cat("Type:", class(x_test1), "\n")
  cat("Names:", names(x_test1), "\n")
}, error = function(e) {
  cat("Error in format 1:", e$message, "\n")
})

cat("\n=== Testing Input Format 2: Unnamed List ===\n")
tryCatch({
  x_test2 <- list(
    array(as.integer(user_indices), dim = c(length(user_indices), 1)),
    array(as.integer(item_indices), dim = c(length(item_indices), 1))
  )
  cat("Format 2 created successfully\n")
  cat("Type:", class(x_test2), "\n")
  cat("Length:", length(x_test2), "\n")
}, error = function(e) {
  cat("Error in format 2:", e$message, "\n")
})

cat("\n=== Testing Input Format 3: Separate Arrays ===\n")
tryCatch({
  user_array <- array(as.integer(user_indices), dim = c(length(user_indices), 1))
  item_array <- array(as.integer(item_indices), dim = c(length(item_indices), 1))
  cat("Format 3 created successfully\n")
  cat("User array shape:", dim(user_array), "\n")
  cat("Item array shape:", dim(item_array), "\n")
}, error = function(e) {
  cat("Error in format 3:", e$message, "\n")
})

# Test with a simple model
cat("\n=== Testing Simple Model ===\n")
tryCatch({
  # Build simple model
  user_input <- layer_input(shape = 1, name = "user_input")
  item_input <- layer_input(shape = 1, name = "item_input")
  
  user_embedding <- user_input %>%
    layer_embedding(input_dim = n_users, output_dim = 8, name = "user_embedding") %>%
    layer_flatten(name = "user_flatten")
  
  item_embedding <- item_input %>%
    layer_embedding(input_dim = n_items, output_dim = 8, name = "item_embedding") %>%
    layer_flatten(name = "item_flatten")
  
  concat <- layer_concatenate(c(user_embedding, item_embedding), name = "concat")
  output <- concat %>%
    layer_dense(units = 1, activation = "linear", name = "output")
  
  model <- keras_model(inputs = c(user_input, item_input), outputs = output)
  
  model$compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    metrics = list("mae")
  )
  
  cat("Model built and compiled successfully\n")
  
  # Test training with different formats
  cat("\n--- Testing training with Format 1 (named list) ---\n")
  tryCatch({
    history1 <- model$fit(
      x = x_test1,
      y = as.numeric(ratings),
      batch_size = 10,
      epochs = 1,
      verbose = 0
    )
    cat("Format 1 training: SUCCESS\n")
  }, error = function(e) {
    cat("Format 1 training ERROR:", e$message, "\n")
  })
  
  cat("\n--- Testing training with Format 2 (unnamed list) ---\n")
  tryCatch({
    history2 <- model$fit(
      x = x_test2,
      y = as.numeric(ratings),
      batch_size = 10,
      epochs = 1,
      verbose = 0
    )
    cat("Format 2 training: SUCCESS\n")
  }, error = function(e) {
    cat("Format 2 training ERROR:", e$message, "\n")
  })
  
  cat("\n--- Testing training with Format 3 (separate arrays) ---\n")
  tryCatch({
    history3 <- model$fit(
      x = list(user_array, item_array),
      y = as.numeric(ratings),
      batch_size = 10,
      epochs = 1,
      verbose = 0
    )
    cat("Format 3 training: SUCCESS\n")
  }, error = function(e) {
    cat("Format 3 training ERROR:", e$message, "\n")
  })
  
}, error = function(e) {
  cat("Model building ERROR:", e$message, "\n")
})
