library(keras)
library(kerastuneR)
library(tensorflow)
library(dplyr)
library(tidyr)

data = read.csv('/Users/ezragoliath/Desktop/DS4I/Assignment 1/data/scotland_avalanche_forecasts_2009_2025.csv', 
                header = T)
# this is a very general solution to the problem of missing entries
data = drop_na(data)
data = filter(data, FAH != '', OAH != '', Precip.Code != '')
## data description ##

# Date = the date that the forecast was made
# Area = one of six forecasting region
# FAH = the forecast avalanche hazard for the following da
# OAH = the observed avalanche hazard on the following day (observation made the following day)
# longitude:Incline: position and topography at forecast location (predictor set)
# Air.Temp:Summit.Wind.Speed = weather in the vicinity of the forecast location at the
# time the forecast was made (predictor set 2)
# Max.Temp.Grad:Snow.Temp = results of a ”snow pack test” of the integrity of snow at
# the forecast location (predictor set 3)

unique(data['Area'])
unique(data['Obs'])
unique(data['FAH'])
unique(data['OAH'])
unique(data['Precip.Code'])


data$FAH         = as.integer(factor(x = data$FAH)) - 1
data$OAH         = as.integer(factor(x = data$OAH)) - 1
data$Precip.Code = as.integer(factor(data$Precip.Code)) - 1

predictor_set_1 = select(data, c('longitude':'Incline'))
predictor_set_1 = mutate(predictor_set_1, across(c(longitude : Incline), scale))
predictor_set_1 = as.matrix(predictor_set_1)

predictor_set_2 = select(data, c('Air.Temp':'Summit.Wind.Speed', 'FAH'))
predictor_set_2 = mutate(predictor_set_2, across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2 = as.matrix(predictor_set_2)

predictor_set_3 = select(data, c('Max.Temp.Grad':'Snow.Temp', 'FAH'))
predictor_set_3 = mutate(predictor_set_3, across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3 = as.matrix(predictor_set_3)

y = data$OAH # or FAH
y = to_categorical(y, num_classes = 5)

set.seed(2025)
training_indices = runif(n = floor(nrow(data) * 0.7), min = 1, max = nrow(data))

predictor_set_1_train = predictor_set_1[training_indices, ] 
predictor_set_1_test  = predictor_set_1[-training_indices, ]

predictor_set_2_train = predictor_set_2[training_indices, ] 
predictor_set_2_test  = predictor_set_2[-training_indices, ]

predictor_set_3_train = predictor_set_3[training_indices, ] 
predictor_set_3_test  = predictor_set_3[-training_indices, ]

y_train = y[training_indices, ]
y_test  = y[-training_indices, ]

set.seed(2025)

input = layer_input(shape = c(10))

output = input %>% 
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(rate = 0.5, seed = 2025) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.5, seed = 2026) %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dropout(rate = 0.5, seed = 2027) %>%
  layer_dense(units = 5, activation = 'softmax')

model = keras_model(inputs = input, outputs = output)

model %>% compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer_adam(learning_rate = 0.005),
                  metrics = c('accuracy'))
  
history = model %>% fit(predictor_set_3_train,
                        y_train,
                        epochs = 100, batch_size = 5, 
                        validation_split = 0.2, shuffle = TRUE)     

plot(history)
     
###

y = data$FAH # or FAH
y = to_categorical(y, num_classes = 5)

y_train = y[training_indices, ]
y_test  = y[-training_indices, ]

set.seed(2025)

input = layer_input(shape = c(10))

output = input %>% 
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 5, activation = 'softmax')

model = keras_model(inputs = input, outputs = output)

model %>% compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer_adam(learning_rate = 0.005),
                  metrics = c('accuracy'))

history = model %>% fit(predictor_set_3_train,
                        y_train,
                        epochs = 100, batch_size = 5, 
                        validation_split = 0.2, shuffle = TRUE)     

plot(history)


#hp = HyperParameters()
#hp$Choice('learning_rate', c(1e-1, 1e-3))
#hp$Int('number_of_layers', 2L, 5L)

model_builder = function(hp){
  
  n_layers = hp$Int('number_of_layers',
                     min_value = 2, max_value = 5, step = 1)
  
  input = layer_input(shape = c(10))
  
  #output = input %>% 
  #  layer_dense(units = 100, activation = 'relu') %>%
  #  layer_dropout(rate = 0.5) %>%
  #  layer_dense(units = 50, activation = 'relu') %>%
  #  layer_dropout(rate = 0.5) %>%
  #  layer_dense(units = 10, activation = 'relu') %>%
  #  layer_dropout(rate = 0.5) %>%
  #  layer_dense(units = 5, activation = 'softmax')
  
  x = input
  for (i in 1:n_layers){
    x = x %>%
      layer_dense(units = 20, activation = 'relu') %>%
      layer_dropout(rate = 0.5)
  }
  output = x %>% 
    layer_dense(units = 5, activation = 'softmax')
  
  model = keras_model(inputs = input, outputs = output)
  
  lr = hp$Choice("learning_rate", values = c(1e-1, 1e-2, 1e-3))
  
  model %>% compile(loss = 'categorical_crossentropy', 
                    optimizer = optimizer_adam(learning_rate = lr),
                    metrics = c('accuracy'))
  
  return(model)
}

tuner = kerastuneR::RandomSearch(hypermodel = model_builder,
                                 objective = "val_accuracy",
                                 max_trials = 50, executions_per_trial = 1,
                                 directory = "tuning_dir",
                                 project_name = "layers_lr_search",
                                 overwrite = TRUE)

tuner %>% fit_tuner(x = predictor_set_3_train,
                    y = y_train,
                    epochs = 50,
                    validation_split = 0.2, shuffle = TRUE)

###__________________________________________________________________________###
 
model = keras_model_sequential() |>
  layer_flatten(input_shape = c(28, 28)) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 10, activation = "softmax")
          
model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

history = model %>% fit(
  x_train, y_train,
  epochs = 3, batch_size = 128,
  validation_split = 0.2
)
