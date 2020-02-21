#Exploring tidymodels library following Julia Silge tutorial
#https://juliasilge.com/blog/hotels-recipes/

library(tidyverse)
library(dplyr)

hotels <- read.csv("~/Desktop/Data Science/kaggle/competitions/hotel booking data set/hotel_bookings.csv")

hotel_stays <- hotels %>%
  filter(is_canceled == 0) %>%
  mutate(
    children = case_when(
      children + babies > 0 ~ "children",
      TRUE ~ "none"
    ),
    required_car_parking_spaces = case_when(
      required_car_parking_spaces > 0 ~ "parking",
      TRUE ~ "none"
    )
  ) %>%
  select(-is_canceled, -reservation_status, -babies)

hotel_stays %>%
  count(children)
library(skimr)

skim(hotel_stays)

# City hotel VS Resort Hotels in terms of children
hotel_stays %>%
  mutate(arrival_date_month = factor(arrival_date_month,
                                     levels = month.name
  )) %>%
  count(hotel, arrival_date_month, children) %>%
  group_by(hotel, children) %>%
  mutate(proportion = n / sum(n)) %>%
  ggplot(aes(arrival_date_month, proportion, fill = children)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~hotel, nrow = 2) +
  labs(
    x = NULL,
    y = "Proportion of hotel stays",
    fill = NULL
  )

#Parking spaces in terms of children
hotel_stays %>%
  count(hotel, required_car_parking_spaces, children) %>%
  group_by(hotel, children) %>%
  mutate(proportion = n / sum(n)) %>%
  ggplot(aes(required_car_parking_spaces, proportion, fill = children)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~hotel, nrow = 2) +
  labs(
    x = NULL,
    y = "Proportion of hotel stays",
    fill = NULL
  )

library(GGally)

hotel_stays %>%
  dplyr::select(
    children, adr,
    required_car_parking_spaces,
    total_of_special_requests
  )


#Feature engeneering
hotels_df <- hotel_stays %>%
  select(
    children, hotel, arrival_date_month, meal, adr, adults,
    required_car_parking_spaces, total_of_special_requests,
    stays_in_week_nights, stays_in_weekend_nights
  ) %>%
  mutate_if(is.character, factor)

library(tidymodels)

set.seed(1234)
hotel_split <- initial_split(hotels_df)

hotel_train <- training(hotel_split)
hotel_test <- testing(hotel_split)

hotel_rec <- recipe(children ~ ., data = hotel_train) %>%
  step_downsample(children) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_numeric()) %>%
  step_normalize(all_numeric()) %>%
  prep()

hotel_rec

test_proc <- bake(hotel_rec, new_data = hotel_test)

#Knn model
knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_fit <- knn_spec %>%
  fit(children ~ ., data = juice(hotel_rec))

knn_fit

#Decision tree model
tree_spec <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")
tree_fit <- tree_spec %>%
  fit(children ~ ., data = juice(hotel_rec))

tree_fit
set.seed(1234)
validation_splits <- mc_cv(juice(hotel_rec), prop = 0.9, strata = children)
validation_splits

#Results
knn_res <- fit_resamples(
  children ~ .,
  knn_spec,
  validation_splits,
  control = control_resamples(save_pred = TRUE)
)

knn_res %>%
  collect_metrics()

tree_res <- fit_resamples(
  children ~ .,
  tree_spec,
  validation_splits,
  control = control_resamples(save_pred = TRUE)
)

tree_res %>%
  collect_metrics()

#Visualization of models results
knn_res %>%
  unnest(.predictions) %>%
  mutate(model = "kknn") %>%
  bind_rows(tree_res %>%
              unnest(.predictions) %>%
              mutate(model = "rpart")) %>%
  group_by(model) %>%
  roc_curve(children, .pred_children) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(size = 1.5) +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )
#Confusion matrix
knn_conf <- knn_res %>%
  unnest(.predictions) %>%
  conf_mat(children, .pred_class)

knn_conf
knn_conf %>%
  autoplot()

knn_fit %>%
  predict(new_data = test_proc, type = "prob") %>%
  mutate(truth = hotel_test$children) %>%
  roc_auc(truth, .pred_children)


