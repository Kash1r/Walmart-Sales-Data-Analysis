
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(lubridate)

# Read the data
walmart_data <- read.csv("C:\Users\Kashir\Desktop\Walmart")

# Convert Date to Date type
walmart_data$Date <- as.Date(walmart_data$Date, format="%d-%m-%Y")

# Trend Analysis
# Aggregating sales data by date
sales_trend <- walmart_data %>%
  group_by(Date) %>%
  summarize(Total_Sales = sum(Weekly_Sales))

# Plotting the time series of sales
ggplot(sales_trend, aes(x = Date, y = Total_Sales)) +
  geom_line() +
  labs(title = "Trend of Total Sales Over Time",
       x = "Date",
       y = "Total Sales")

# Seasonality Analysis
# Creating a column to indicate whether the week is a holiday week
# Assuming holiday dates are as provided
super_bowl_dates <- as.Date(c("2010-02-12", "2011-02-11", "2012-02-10"))
labour_day_dates <- as.Date(c("2010-09-10", "2011-09-09", "2012-09-07"))
thanksgiving_dates <- as.Date(c("2010-11-26", "2011-11-25", "2012-11-23"))
christmas_dates <- as.Date(c("2010-12-31", "2011-12-30", "2012-12-28"))

holiday_dates <- c(super_bowl_dates, labour_day_dates, thanksgiving_dates, christmas_dates)
walmart_data$Is_Holiday_Week <- as.integer(walmart_data$Date %in% holiday_dates)

# Comparing Sales During Holiday Weeks vs. Non-Holiday Weeks
holiday_sales_comparison <- walmart_data %>%
  group_by(Is_Holiday_Week) %>%
  summarize(Average_Sales = mean(Weekly_Sales))

# Visualizing the comparison
ggplot(holiday_sales_comparison, aes(x = factor(Is_Holiday_Week), y = Average_Sales, fill = factor(Is_Holiday_Week))) +
  geom_bar(stat="identity") +
  labs(title = "Comparison of Average Sales: Holiday Weeks vs. Non-Holiday Weeks",
       x = "Is Holiday Week",
       y = "Average Sales")

