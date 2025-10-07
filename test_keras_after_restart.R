# Test Keras After Restart
library(keras)
library(tensorflow)
library(dplyr)
library(tidyr)

# Load data
load("book_ratings.Rdata")

# Create user-item matrix function
create_user_item_matrix <- function(ratings_data, min_ratings_per_book = 3, min_ratings_per_user = 3) {
  ratings_clean <- ratings_data %>%
    mutate(Book.Rating = ifelse(Book.Rating == 0, NA, Book.Rating))
  
  user_item_matrix <- ratings_clean %>%
    select(User.ID, ISBN, Book.Rating) %>%
    pivot_wider(names_from = ISBN, values_from = Book.Rating, values_fill = NA)
  
  user_ids <- user_item_matrix$User.ID
  user_item_matrix <- as.matrix(user_item_matrix[, -1])
  rownames(user_item_matrix) <- user_ids
  
  books_to_keep <- colSums(!is.na(user_item_matrix)) >= min_ratings_per_book
  user_item_matrix <- user_item_matrix[, books_to_keep]
  cat("Kept", sum(books_to_keep), "books with >=", min_ratings_per_book, "ratings\n")
  
  users_to_keep <- rowSums(!is.na(user_item_matrix)) >= min_ratings_per_user
  user_item_matrix <- user_item_matrix[users_to_keep, ]
  cat("Kept", sum(users_to_keep), "users with >=", min_ratings_per_user, "ratings\n")
  
  cat("Final matrix:", nrow(user_item_matrix), "users x", ncol(user_item_matrix), "books\n\n")
  
  return(user_item_matrix)
}

# Merge data
data <- book_ratings %>%
  left_join(book_info, by = "ISBN")

# Create user-item matrix
user_item_matrix_nn <- create_user_item_matrix(
  data, 
  min_ratings_per_book = 5,
  min_ratings_per_user = 2
)

# Test data preparation
prepare_nn_data <- function(user_item_matrix, test_ratio = 0.2, seed = 123) {
  
  set.seed(seed)
  
  # Get observed ratings
  observed <- which(!is.na(user_item_matrix), arr.ind = TRUE)
  n_ratings <- nrow(observed)
  n_users <- nrow(user_item_matrix)
  n_items <- ncol(user_item_matrix)
  
  if (n_ratings < 3) {
    stop("Not enough ratings for train/test split (need at least 3 ratings)")
  }
  
  # Random sampling for train/test split
  test_size <- max(1, floor(n_ratings * test_ratio))
  test_indices <- sample(1:n_ratings, size = test_size)
  train_indices <- setdiff(1:n_ratings, test_indices)
  
  # Split data
  test_obs <- observed[test_indices, , drop = FALSE]
  train_obs <- observed[train_indices, , drop = FALSE]
  
  # Create training data (0-based indexing for neural network) with proper data types
  train_data <- data.frame(
    user_index = as.integer(train_obs[, 1] - 1),
    item_index = as.integer(train_obs[, 2] - 1),
    rating = as.numeric(user_item_matrix[train_obs])
  )
  
  test_data <- data.frame(
    user_index = as.integer(test_obs[, 1] - 1),
    item_index = as.integer(test_obs[, 2] - 1),
    rating = as.numeric(user_item_matrix[test_obs])
  )
  
  cat("Training samples:", nrow(train_data), "\n")
  cat("Test samples:", nrow(test_data), "\n")
  cat("Users:", n_users, "Items:", n_items, "\n")
  
  return(list(
    train = train_data,
    test = test_data,
    n_users = n_users,
    n_items = n_items
  ))
}

# Test the build_nn_model function with all fixes
build_nn_model <- function(n_users, n_items, embedding_dim = 50, dense_units = 128, dropout_rate = 0.3) {
  
  # User embedding
  user_input <- layer_input(shape = 1, name = "user_input")
  user_embedding <- user_input %>%
    layer_embedding(input_dim = n_users, output_dim = embedding_dim, name = "user_embedding") %>%
    layer_flatten(name = "user_flatten")
  
  # Item embedding
  item_input <- layer_input(shape = 1, name = "item_input")
  item_embedding <- item_input %>%
    layer_embedding(input_dim = n_items, output_dim = embedding_dim, name = "item_embedding") %>%
    layer_flatten(name = "item_flatten")
  
  # Concatenate embeddings
  concat <- layer_concatenate(c(user_embedding, item_embedding), name = "concat")
  
  # Dense layers
  dense1 <- concat %>%
    layer_dense(units = dense_units, activation = "relu", name = "dense1") %>%
    layer_dropout(rate = dropout_rate, name = "dropout1")
  
  dense2 <- dense1 %>%
    layer_dense(units = dense_units/2, activation = "relu", name = "dense2") %>%
    layer_dropout(rate = dropout_rate, name = "dropout2")
  
  # Output layer (rating prediction)
  output <- dense2 %>%
    layer_dense(units = 1, activation = "linear", name = "output")
  
  # Create model
  model <- keras_model(inputs = c(user_input, item_input), outputs = output)
  
  # Compile model - FIXED VERSION
  model$compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse",
    metrics = list("mae")
  )
  
  return(model)
}

# Test the train_nn_model function with all fixes
train_nn_model <- function(train_data, n_users, n_items, 
                          embedding_dim = 50, dense_units = 128, dropout_rate = 0.3,
                          batch_size = 256, epochs = 50, validation_split = 0.1,
                          verbose = 1) {
  
  # Build model
  model <- build_nn_model(n_users, n_items, embedding_dim, dense_units, dropout_rate)
  
  # Prepare training data with proper data types (numpy arrays)
  x_train <- list(
    user_input = array(as.integer(train_data$user_index), dim = c(length(train_data$user_index), 1)),
    item_input = array(as.integer(train_data$item_index), dim = c(length(train_data$item_index), 1))
  )
  y_train <- as.numeric(train_data$rating)
  
  # Train model - FIXED VERSION
  history <- model$fit(
    x = x_train,
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = validation_split,
    verbose = verbose,
    callbacks = list(
      callback_early_stopping(patience = 5, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(patience = 3, factor = 0.5)
    )
  )
  
  return(list(model = model, history = history))
}

# Test data preparation
cat("Testing data preparation...\n")
nn_data <- prepare_nn_data(user_item_matrix_nn, test_ratio = 0.2, seed = 123)

# Test model building
cat("\nTesting model building...\n")
test_model <- build_nn_model(
  n_users = nn_data$n_users,
  n_items = nn_data$n_items,
  embedding_dim = 16,
  dense_units = 32,
  dropout_rate = 0.3
)
cat("Model built successfully!\n")

# Test model training (with very small parameters for quick test)
cat("\nTesting model training...\n")
model_result <- train_nn_model(
  train_data = nn_data$train,
  n_users = nn_data$n_users,
  n_items = nn_data$n_items,
  embedding_dim = 8,      # Very small for quick test
  dense_units = 16,       # Very small for quick test
  dropout_rate = 0.3,
  batch_size = 64,
  epochs = 2,             # Very few epochs for quick test
  verbose = 1
)

cat("Model trained successfully!\n")
cat("Training completed without errors!\n")
cat("The Keras fix is working properly!\n")

