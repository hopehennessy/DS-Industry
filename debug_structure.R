library(recosystem)
library(dplyr)
library(tidyr)
library(tibble)

# Load data
load('book_ratings.Rdata')

# Set seed for reproducibility
set.seed(123)

# Create user-item matrix (simplified version)
create_user_item_matrix <- function(data, min_ratings_per_user = 3, min_ratings_per_item = 3) {
  # Count ratings per user and item
  user_counts <- data %>% group_by(User.ID) %>% summarise(n = n())
  item_counts <- data %>% group_by(ISBN) %>% summarise(n = n())
  
  # Filter users and items
  valid_users <- user_counts$User.ID[user_counts$n >= min_ratings_per_user]
  valid_items <- item_counts$ISBN[item_counts$n >= min_ratings_per_item]
  
  # Filter data
  filtered_data <- data %>% 
    filter(User.ID %in% valid_users, ISBN %in% valid_items)
  
  # Create matrix
  user_item_matrix <- filtered_data %>%
    pivot_wider(names_from = ISBN, values_from = Book.Rating, values_fill = NA) %>%
    column_to_rownames("User.ID") %>%
    as.matrix()
  
  return(user_item_matrix)
}

# Prepare data function
prepare_recosystem_data_improved <- function(user_item_matrix) {
  # Convert matrix to long format with observed ratings only
  observed <- which(!is.na(user_item_matrix), arr.ind = TRUE)
  
  # Create training data with 0-based indexing 
  train_data <- data.frame(
    user_index = observed[, 1] - 1,
    item_index = observed[, 2] - 1,
    rating = user_item_matrix[observed]  # Use raw ratings
  )
  
  # Store ID mappings
  user_ids <- data.frame(
    user_index = 0:(nrow(user_item_matrix) - 1),
    user_id = rownames(user_item_matrix)
  )
  
  item_ids <- data.frame(
    item_index = 0:(ncol(user_item_matrix) - 1),
    item_id = colnames(user_item_matrix)
  )
  
  return(list(
    train_data = train_data,
    user_ids = user_ids,
    item_ids = item_ids,
    n_users = nrow(user_item_matrix),
    n_items = ncol(user_item_matrix)
  ))
}

# Create user-item matrix
user_item_matrix <- create_user_item_matrix(book_ratings, min_ratings_per_user = 3, min_ratings_per_item = 3)

# Prepare data
prepared <- prepare_recosystem_data_improved(user_item_matrix)

# Create data source
train_set <- data_memory(
  user_index = prepared$train_data$user_index,
  item_index = prepared$train_data$item_index,
  rating = prepared$train_data$rating,
  index1 = FALSE
)

# Create model and tune
r <- Reco()
r$tune(train_set, opts = list(
  dim = c(10, 20, 30),
  lrate = c(0.1, 0.05, 0.01),
  costp_l2 = c(0.01, 0.1),
  costq_l2 = c(0.01, 0.1),
  niter = 50,
  nthread = 4,
  verbose = FALSE
))

# Debug: Check what's in train_pars
cat("Structure of r$train_pars:\n")
str(r$train_pars)

cat("\nContents of r$train_pars:\n")
print(r$train_pars)

cat("\nContents of r$train_pars$min:\n")
print(r$train_pars$min)

cat("\nContents of r$train_pars$res:\n")
print(head(r$train_pars$res))

# Find the best parameters manually
if (!is.null(r$train_pars$res)) {
  best_idx <- which.min(r$train_pars$res$loss_fun)
  best_params <- r$train_pars$res[best_idx, ]
  cat("\nBest parameters from res table:\n")
  print(best_params)
}