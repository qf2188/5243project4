library(shiny)
library(shinydashboard)
library(dplyr)
library(ggplot2)
library(DT)
library(readr)
library(scales)

# -----------------------------
# 1. Data Loading Utilities
# -----------------------------

safe_read_csv <- function(paths) {
  for (p in paths) {
    if (file.exists(p)) {
      message("Loading data from: ", p)
      return(readr::read_csv(p, show_col_types = FALSE))
    }
  }
  return(NULL)
}

standardize_names <- function(df) {
  names(df) <- toupper(names(df))
  names(df) <- gsub("\\.", "_", names(df))
  df
}

make_demo_data <- function(n = 8000, seed = 5243) {
  set.seed(seed)
  carriers <- c("AA", "AS", "B6", "DL", "F9", "G4", "HA", "MQ", "NK", "OH", "OO", "UA", "WN", "YX")
  airports <- c("ATL", "BOS", "CLT", "DCA", "DEN", "DFW", "EWR", "JFK", "LAS", "LAX", "LGA", "MIA", "ORD", "PHX", "SEA", "SFO")
  origin <- sample(airports, n, replace = TRUE)
  dest <- sample(airports, n, replace = TRUE)
  same <- origin == dest
  while (any(same)) {
    dest[same] <- sample(airports, sum(same), replace = TRUE)
    same <- origin == dest
  }
  dep_hour <- sample(0:23, n, replace = TRUE, prob = c(0.02,0.01,0.01,0.01,0.02,0.05,0.06,0.06,0.05,0.05,0.05,0.05,
                                                       0.05,0.05,0.05,0.05,0.05,0.05,0.06,0.06,0.05,0.04,0.03,0.02))
  dow <- sample(1:7, n, replace = TRUE)
  distance <- round(pmax(80, rlnorm(n, log(750), 0.65)))
  dep_delay <- round(rnorm(n, mean = ifelse(dep_hour >= 16 & dep_hour <= 21, 12, 5), sd = 22))
  dep_delay <- pmax(dep_delay, -25)
  carrier_effect <- ifelse(sample(carriers, n, replace = TRUE) %in% c("B6", "F9", "G4", "OH", "OO"), 0.45, 0)
  carrier <- sample(carriers, n, replace = TRUE)
  evening_effect <- ifelse(dep_hour >= 15 & dep_hour <= 21, 0.55, 0)
  weekend_effect <- ifelse(dow %in% c(6, 7), 0.15, 0)
  dep_effect <- 0.055 * dep_delay
  airport_effect <- ifelse(origin %in% c("EWR", "LGA", "ORD", "SFO"), 0.35, 0)
  logit <- -2.55 + dep_effect + evening_effect + weekend_effect + airport_effect + ifelse(carrier %in% c("B6", "F9", "G4", "OH", "OO"), 0.35, 0)
  prob <- 1 / (1 + exp(-logit))
  arr_del15 <- rbinom(n, 1, prob)
  arr_delay <- round(ifelse(arr_del15 == 1, rgamma(n, 2.2, 1/22), rnorm(n, -7, 9)))
  cancelled <- rbinom(n, 1, 0.018)
  diverted <- rbinom(n, 1, 0.003)
  pred_prob <- pmin(pmax(prob + rnorm(n, 0, 0.035), 0.01), 0.99)

  tibble(
    CARRIER = carrier,
    ORIGIN = origin,
    DEST = dest,
    DEP_HOUR = dep_hour,
    DAY_OF_WEEK = dow,
    IS_WEEKEND = ifelse(dow %in% c(6,7), 1, 0),
    DISTANCE = distance,
    DISTANCE_BUCKET = cut(distance, breaks = c(0, 500, 1000, 2000, Inf),
                          labels = c("Short", "Medium", "Long", "Very long"), right = FALSE),
    DEP_DELAY = dep_delay,
    ARR_DELAY = arr_delay,
    ARR_DEL15 = arr_del15,
    CANCELLED = cancelled,
    DIVERTED = diverted,
    PRED_PROB = pred_prob,
    PRED_CLASS = ifelse(pred_prob >= 0.5, 1, 0)
  )
}

raw_df <- safe_read_csv(c(
  "data/final_model_predictions.csv",
  "data/cleaned_flight_data.csv",
  "data/processed_flight_data.csv",
  "final_model_predictions.csv",
  "cleaned_flight_data.csv",
  "processed_flight_data.csv"
))

if (is.null(raw_df)) {
  flight_df <- make_demo_data()
  data_note <- "Demo data are being displayed because no project CSV was found. Place final_model_predictions.csv or cleaned_flight_data.csv in the data/ folder to use real project data."
} else {
  flight_df <- standardize_names(raw_df)
  data_note <- "Project data loaded from local CSV."
}

# Create compatible columns if names differ across files
if (!"CARRIER" %in% names(flight_df) && "OP_UNIQUE_CARRIER" %in% names(flight_df)) flight_df$CARRIER <- flight_df$OP_UNIQUE_CARRIER
if (!"ORIGIN" %in% names(flight_df) && "ORIGIN_AIRPORT_ID" %in% names(flight_df)) flight_df$ORIGIN <- as.character(flight_df$ORIGIN_AIRPORT_ID)
if (!"DEST" %in% names(flight_df) && "DEST_AIRPORT_ID" %in% names(flight_df)) flight_df$DEST <- as.character(flight_df$DEST_AIRPORT_ID)
if (!"DEP_HOUR" %in% names(flight_df) && "CRS_DEP_TIME" %in% names(flight_df)) flight_df$DEP_HOUR <- floor(as.numeric(flight_df$CRS_DEP_TIME) / 100)
if (!"DAY_OF_WEEK" %in% names(flight_df)) flight_df$DAY_OF_WEEK <- NA_integer_
if (!"IS_WEEKEND" %in% names(flight_df) && "DAY_OF_WEEK" %in% names(flight_df)) flight_df$IS_WEEKEND <- ifelse(flight_df$DAY_OF_WEEK %in% c(6,7), 1, 0)
if (!"DISTANCE_BUCKET" %in% names(flight_df) && "DISTANCE" %in% names(flight_df)) {
  flight_df$DISTANCE_BUCKET <- cut(as.numeric(flight_df$DISTANCE), breaks = c(0, 500, 1000, 2000, Inf),
                                   labels = c("Short", "Medium", "Long", "Very long"), right = FALSE)
}
if (!"ARR_DEL15" %in% names(flight_df) && "ARR_DELAY" %in% names(flight_df)) flight_df$ARR_DEL15 <- ifelse(as.numeric(flight_df$ARR_DELAY) > 15, 1, 0)
if (!"CANCELLED" %in% names(flight_df)) flight_df$CANCELLED <- 0
if (!"DIVERTED" %in% names(flight_df)) flight_df$DIVERTED <- 0
if (!"PRED_PROB" %in% names(flight_df)) flight_df$PRED_PROB <- NA_real_

# Keep app responsive for very large files
set.seed(5243)
if (nrow(flight_df) > 100000) {
  app_df <- flight_df %>% sample_n(100000)
  data_note <- paste0(data_note, " For app responsiveness, a random sample of 100,000 rows is displayed.")
} else {
  app_df <- flight_df
}

app_df <- app_df %>%
  mutate(
    CARRIER = as.character(CARRIER),
    ORIGIN = as.character(ORIGIN),
    DEST = as.character(DEST),
    ARR_DEL15 = as.integer(ARR_DEL15),
    CANCELLED = as.integer(CANCELLED),
    DIVERTED = as.integer(DIVERTED),
    DEP_HOUR = suppressWarnings(as.integer(DEP_HOUR)),
    DISTANCE = suppressWarnings(as.numeric(DISTANCE)),
    DEP_DELAY = suppressWarnings(as.numeric(DEP_DELAY)),
    ARR_DELAY = suppressWarnings(as.numeric(ARR_DELAY))
  )

model_metrics <- tibble(
  Model = c("Logistic Regression", "Random Forest", "XGBoost", "Tuned XGBoost"),
  Accuracy = c(0.8985, 0.8977, 0.9269, 0.9277),
  Precision = c(0.7180, 0.7238, 0.9084, 0.9105),
  Recall = c(0.7996, 0.7845, 0.7002, 0.7024),
  F1 = c(0.7566, 0.7522, 0.7908, 0.7930),
  ROC_AUC = c(0.9215, 0.8813, 0.9312, 0.9373),
  Runtime_Seconds = c(79.9603, 6.1764, 7.8659, 2.0720)
)

# -----------------------------
# 2. UI
# -----------------------------

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Flight Delay Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("plane")),
      menuItem("Explore Delay Patterns", tabName = "eda", icon = icon("chart-column")),
      menuItem("Compare Models", tabName = "models", icon = icon("gauge-high")),
      menuItem("Predict Delay Risk", tabName = "predict", icon = icon("wand-magic-sparkles")),
      menuItem("Interpretation", tabName = "interpret", icon = icon("circle-info"))
    )
  ),
  dashboardBody(
    tags$head(tags$style(HTML("\n      .small-box { border-radius: 12px; }\n      .box { border-radius: 12px; }\n      .content-wrapper { background-color: #f7f8fa; }\n      .note-box { background:#eef6ff; border-left:5px solid #2c7fb8; padding:12px; border-radius:8px; margin-bottom:12px; }\n      .warning-box { background:#fff7e6; border-left:5px solid #f0ad4e; padding:12px; border-radius:8px; margin-bottom:12px; }\n    "))),
    tabItems(
      tabItem(
        tabName = "overview",
        fluidRow(
          box(width = 12, title = "Project Question", status = "primary", solidHeader = TRUE,
              h3("Can we predict whether a U.S. domestic flight will arrive more than 15 minutes late?"),
              p("This Shiny dashboard is an optional enhancement for STAT 5243 Project 4. It translates the flight-delay analysis into an interactive decision-support prototype."),
              div(class = "note-box", strong("Data note: "), data_note),
              p(strong("Data source: "), "Bureau of Transportation Statistics Airline On-Time Performance data."),
              p(strong("Modeling target: "), "ARR_DEL15, where 1 means the flight arrived more than 15 minutes late."),
              p(strong("Final selected model: "), "Tuned XGBoost classifier."),
              p(strong("Repository: "), tags$a(href = "https://github.com/qf2188/5243project4/tree/main", target = "_blank", "GitHub Project Repository"))
          )
        ),
        fluidRow(
          valueBoxOutput("totalFlights", width = 3),
          valueBoxOutput("delayRate", width = 3),
          valueBoxOutput("cancelRate", width = 3),
          valueBoxOutput("avgArrDelay", width = 3)
        ),
        fluidRow(
          box(width = 6, title = "Final Model Performance", status = "primary", solidHeader = TRUE,
              tableOutput("finalMetrics")),
          box(width = 6, title = "Dashboard Components", status = "primary", solidHeader = TRUE,
              tags$ul(
                tags$li("Explore delay patterns by carrier, airport, time, and distance group."),
                tags$li("Compare logistic regression, random forest, XGBoost, and tuned XGBoost."),
                tags$li("Interactively adjust classification threshold and inspect precision/recall tradeoffs."),
                tags$li("Test a simplified delay-risk prediction prototype."
                )
              ))
        )
      ),

      tabItem(
        tabName = "eda",
        fluidRow(
          box(width = 3, title = "Filters", status = "primary", solidHeader = TRUE,
              selectInput("carrierFilter", "Carrier", choices = c("All", sort(unique(na.omit(app_df$CARRIER)))), selected = "All"),
              selectInput("originFilter", "Origin Airport", choices = c("All", sort(unique(na.omit(app_df$ORIGIN)))), selected = "All"),
              sliderInput("hourFilter", "Scheduled Departure Hour", min = 0, max = 23, value = c(0, 23), step = 1),
              selectInput("weekendFilter", "Day Type", choices = c("All", "Weekday", "Weekend"), selected = "All"),
              selectInput("distanceFilter", "Distance Group", choices = c("All", sort(unique(na.omit(as.character(app_df$DISTANCE_BUCKET))))), selected = "All")
          ),
          box(width = 9, title = "Filtered Data Summary", status = "primary", solidHeader = TRUE,
              fluidRow(
                valueBoxOutput("filteredN", width = 3),
                valueBoxOutput("filteredDelay", width = 3),
                valueBoxOutput("filteredCancel", width = 3),
                valueBoxOutput("filteredAvgDelay", width = 3)
              ),
              DTOutput("previewTable")
          )
        ),
        fluidRow(
          box(width = 6, title = "Delay Rate by Carrier", status = "info", solidHeader = TRUE, plotOutput("carrierPlot", height = 320)),
          box(width = 6, title = "Delay Rate by Departure Hour", status = "info", solidHeader = TRUE, plotOutput("hourPlot", height = 320))
        ),
        fluidRow(
          box(width = 6, title = "Arrival Delay Distribution", status = "info", solidHeader = TRUE, plotOutput("delayHist", height = 320)),
          box(width = 6, title = "Highest-Delay Origin Airports", status = "info", solidHeader = TRUE, plotOutput("airportPlot", height = 320))
        )
      ),

      tabItem(
        tabName = "models",
        fluidRow(
          box(width = 4, title = "Model Comparison Controls", status = "primary", solidHeader = TRUE,
              selectInput("metricChoice", "Metric to Compare",
                          choices = c("Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Runtime_Seconds"),
                          selected = "F1"),
              sliderInput("threshold", "Classification Threshold", min = 0.05, max = 0.95, value = 0.50, step = 0.05),
              div(class = "warning-box", "The threshold analysis uses PRED_PROB if available. If no prediction-probability column is found, the app displays a demonstration based on the sample data."))
          ,
          box(width = 8, title = "Cross-Validation and Final Model Metrics", status = "primary", solidHeader = TRUE,
              plotOutput("metricPlot", height = 300),
              DTOutput("metricTable"))
        ),
        fluidRow(
          box(width = 6, title = "Threshold-Based Confusion Matrix", status = "info", solidHeader = TRUE,
              tableOutput("confMat")),
          box(width = 6, title = "Threshold-Based Metrics", status = "info", solidHeader = TRUE,
              tableOutput("thresholdMetrics"),
              p("Lowering the threshold usually increases recall but may reduce precision. Raising the threshold usually increases precision but may miss more delayed flights."))
        )
      ),

      tabItem(
        tabName = "predict",
        fluidRow(
          box(width = 4, title = "Flight Inputs", status = "primary", solidHeader = TRUE,
              selectInput("predCarrier", "Carrier", choices = sort(unique(na.omit(app_df$CARRIER)))) ,
              selectInput("predOrigin", "Origin Airport", choices = sort(unique(na.omit(app_df$ORIGIN)))) ,
              selectInput("predDest", "Destination Airport", choices = sort(unique(na.omit(app_df$DEST)))) ,
              sliderInput("predHour", "Scheduled Departure Hour", min = 0, max = 23, value = 17, step = 1),
              selectInput("predDow", "Day of Week", choices = c("Monday"=1,"Tuesday"=2,"Wednesday"=3,"Thursday"=4,"Friday"=5,"Saturday"=6,"Sunday"=7), selected = 5),
              numericInput("predDistance", "Distance in Miles", value = 750, min = 50, max = 5000, step = 50),
              numericInput("predDepDelay", "Departure Delay in Minutes, if available", value = 10, min = -30, max = 500, step = 5),
              sliderInput("predThreshold", "Decision Threshold", min = 0.05, max = 0.95, value = 0.50, step = 0.05)
          ),
          box(width = 8, title = "Delay Risk Output", status = "primary", solidHeader = TRUE,
              h3(textOutput("riskLabel")),
              h2(textOutput("riskProbability")),
              plotOutput("riskGauge", height = 170),
              div(class = "warning-box",
                  strong("Important limitation: "),
                  "This prediction demo is a simplified prototype. Because the final project model includes departure-delay information, it should be interpreted as a near-real-time delay-risk tool rather than a fully pre-departure forecasting system."),
              h4("Explanation"),
              textOutput("riskExplanation")
          )
        )
      ),

      tabItem(
        tabName = "interpret",
        fluidRow(
          box(width = 12, title = "Interpretation and Limitations", status = "primary", solidHeader = TRUE,
              h4("Why XGBoost was selected"),
              p("XGBoost achieved the strongest overall balance across accuracy, precision, F1 score, and ROC-AUC. Logistic regression had higher recall, but XGBoost produced substantially higher precision and stronger probability-ranking performance."),
              h4("Operational interpretation"),
              p("The dashboard should be interpreted as a decision-support prototype. A high predicted probability indicates elevated delay risk, but it does not guarantee a delay. The appropriate decision threshold depends on whether the user cares more about reducing false alarms or capturing more true delays."),
              h4("Major limitation"),
              p("The current model includes departure-delay information. This is useful for near-real-time monitoring after scheduled departure, but a strict pre-departure model should exclude DEP_DELAY and instead rely on schedule, route, carrier, airport, historical congestion, weather, and other information available before departure."),
              h4("Future improvements"),
              tags$ul(
                tags$li("Add weather data from NOAA or another reliable weather source."),
                tags$li("Train on a full year of flight data to capture seasonal patterns."),
                tags$li("Use time-based validation, training on earlier months and testing on later months."),
                tags$li("Add probability calibration and threshold optimization for different operational use cases."),
                tags$li("Deploy a real model object instead of the simplified app-side prediction formula.")
              )
          )
        )
      )
    )
  )
)

# -----------------------------
# 3. Server
# -----------------------------

server <- function(input, output, session) {

  filtered_df <- reactive({
    df <- app_df
    if (!is.null(input$carrierFilter) && input$carrierFilter != "All") df <- df %>% filter(CARRIER == input$carrierFilter)
    if (!is.null(input$originFilter) && input$originFilter != "All") df <- df %>% filter(ORIGIN == input$originFilter)
    if (!is.null(input$hourFilter)) df <- df %>% filter(is.na(DEP_HOUR) | (DEP_HOUR >= input$hourFilter[1] & DEP_HOUR <= input$hourFilter[2]))
    if (!is.null(input$weekendFilter) && input$weekendFilter == "Weekday") df <- df %>% filter(IS_WEEKEND == 0 | is.na(IS_WEEKEND))
    if (!is.null(input$weekendFilter) && input$weekendFilter == "Weekend") df <- df %>% filter(IS_WEEKEND == 1)
    if (!is.null(input$distanceFilter) && input$distanceFilter != "All") df <- df %>% filter(as.character(DISTANCE_BUCKET) == input$distanceFilter)
    df
  })

  delay_rate <- function(df) mean(df$ARR_DEL15 == 1, na.rm = TRUE)
  cancel_rate <- function(df) mean(df$CANCELLED == 1, na.rm = TRUE)
  avg_delay <- function(df) mean(df$ARR_DELAY, na.rm = TRUE)

  output$totalFlights <- renderValueBox({
    valueBox(comma(nrow(app_df)), "Flights in App Data", icon = icon("database"), color = "blue")
  })
  output$delayRate <- renderValueBox({
    valueBox(percent(delay_rate(app_df), accuracy = 0.1), "Arrival Delay Rate", icon = icon("clock"), color = "yellow")
  })
  output$cancelRate <- renderValueBox({
    valueBox(percent(cancel_rate(app_df), accuracy = 0.1), "Cancellation Rate", icon = icon("ban"), color = "red")
  })
  output$avgArrDelay <- renderValueBox({
    valueBox(round(avg_delay(app_df), 2), "Average Arrival Delay", icon = icon("chart-line"), color = "green")
  })

  output$finalMetrics <- renderTable({
    model_metrics %>% filter(Model == "Tuned XGBoost") %>%
      mutate(across(where(is.numeric), ~round(.x, 4)))
  })

  output$filteredN <- renderValueBox({
    valueBox(comma(nrow(filtered_df())), "Filtered Flights", icon = icon("filter"), color = "blue")
  })
  output$filteredDelay <- renderValueBox({
    valueBox(percent(delay_rate(filtered_df()), accuracy = 0.1), "Filtered Delay Rate", icon = icon("clock"), color = "yellow")
  })
  output$filteredCancel <- renderValueBox({
    valueBox(percent(cancel_rate(filtered_df()), accuracy = 0.1), "Filtered Cancellation Rate", icon = icon("ban"), color = "red")
  })
  output$filteredAvgDelay <- renderValueBox({
    valueBox(round(avg_delay(filtered_df()), 2), "Filtered Avg. Arrival Delay", icon = icon("chart-line"), color = "green")
  })

  output$previewTable <- renderDT({
    show_cols <- intersect(c("CARRIER", "ORIGIN", "DEST", "DEP_HOUR", "DAY_OF_WEEK", "DISTANCE", "DEP_DELAY", "ARR_DELAY", "ARR_DEL15", "PRED_PROB"), names(filtered_df()))
    datatable(filtered_df() %>% select(all_of(show_cols)) %>% head(200), options = list(pageLength = 8, scrollX = TRUE))
  })

  output$carrierPlot <- renderPlot({
    df <- filtered_df() %>%
      group_by(CARRIER) %>%
      summarise(delay_rate = mean(ARR_DEL15 == 1, na.rm = TRUE), n = n(), .groups = "drop") %>%
      filter(n >= 20) %>%
      arrange(desc(delay_rate)) %>%
      slice_head(n = 15)
    ggplot(df, aes(x = reorder(CARRIER, delay_rate), y = delay_rate)) +
      geom_col() +
      coord_flip() +
      scale_y_continuous(labels = percent) +
      labs(x = "Carrier", y = "Delay Rate", title = "Top Carriers by Delay Rate") +
      theme_minimal(base_size = 12)
  })

  output$hourPlot <- renderPlot({
    df <- filtered_df() %>%
      filter(!is.na(DEP_HOUR)) %>%
      group_by(DEP_HOUR) %>%
      summarise(delay_rate = mean(ARR_DEL15 == 1, na.rm = TRUE), n = n(), .groups = "drop")
    ggplot(df, aes(x = DEP_HOUR, y = delay_rate)) +
      geom_line(linewidth = 1) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = 0:23) +
      scale_y_continuous(labels = percent) +
      labs(x = "Scheduled Departure Hour", y = "Delay Rate", title = "Delay Rate by Departure Hour") +
      theme_minimal(base_size = 12)
  })

  output$delayHist <- renderPlot({
    df <- filtered_df() %>% filter(!is.na(ARR_DELAY), ARR_DELAY >= -60, ARR_DELAY <= 180)
    ggplot(df, aes(x = ARR_DELAY)) +
      geom_histogram(bins = 45) +
      geom_vline(xintercept = 15, linetype = "dashed", linewidth = 1) +
      labs(x = "Arrival Delay in Minutes", y = "Number of Flights", title = "Arrival Delay Distribution") +
      theme_minimal(base_size = 12)
  })

  output$airportPlot <- renderPlot({
    df <- filtered_df() %>%
      group_by(ORIGIN) %>%
      summarise(delay_rate = mean(ARR_DEL15 == 1, na.rm = TRUE), n = n(), .groups = "drop") %>%
      filter(n >= 20) %>%
      arrange(desc(delay_rate)) %>%
      slice_head(n = 15)
    ggplot(df, aes(x = reorder(ORIGIN, delay_rate), y = delay_rate)) +
      geom_col() +
      coord_flip() +
      scale_y_continuous(labels = percent) +
      labs(x = "Origin Airport", y = "Delay Rate", title = "Highest-Delay Origin Airports") +
      theme_minimal(base_size = 12)
  })

  output$metricPlot <- renderPlot({
    metric <- input$metricChoice
    ggplot(model_metrics, aes(x = reorder(Model, .data[[metric]]), y = .data[[metric]])) +
      geom_col() +
      coord_flip() +
      labs(x = "Model", y = metric, title = paste("Model Comparison by", metric)) +
      theme_minimal(base_size = 12)
  })

  output$metricTable <- renderDT({
    datatable(model_metrics %>% mutate(across(where(is.numeric), ~round(.x, 4))), options = list(pageLength = 5, scrollX = TRUE))
  })

  threshold_df <- reactive({
    df <- app_df
    if (all(is.na(df$PRED_PROB))) {
      # fallback demonstration probability from simplified formula
      dd <- ifelse(is.na(df$DEP_DELAY), 0, df$DEP_DELAY)
      hr <- ifelse(is.na(df$DEP_HOUR), 12, df$DEP_HOUR)
      dist <- ifelse(is.na(df$DISTANCE), 750, df$DISTANCE)
      logit <- -2.6 + 0.055 * dd + ifelse(hr >= 15 & hr <= 21, 0.5, 0) + ifelse(dist > 1500, -0.15, 0)
      df$PRED_PROB <- 1 / (1 + exp(-logit))
    }
    df %>% mutate(PRED_CLASS_APP = ifelse(PRED_PROB >= input$threshold, 1, 0))
  })

  output$confMat <- renderTable({
    df <- threshold_df() %>% filter(!is.na(ARR_DEL15), !is.na(PRED_CLASS_APP))
    tp <- sum(df$ARR_DEL15 == 1 & df$PRED_CLASS_APP == 1, na.rm = TRUE)
    tn <- sum(df$ARR_DEL15 == 0 & df$PRED_CLASS_APP == 0, na.rm = TRUE)
    fp <- sum(df$ARR_DEL15 == 0 & df$PRED_CLASS_APP == 1, na.rm = TRUE)
    fn <- sum(df$ARR_DEL15 == 1 & df$PRED_CLASS_APP == 0, na.rm = TRUE)
    data.frame(
      Actual = c("Not Delayed", "Delayed"),
      `Predicted Not Delayed` = c(tn, fn),
      `Predicted Delayed` = c(fp, tp),
      check.names = FALSE
    )
  })

  output$thresholdMetrics <- renderTable({
    df <- threshold_df() %>% filter(!is.na(ARR_DEL15), !is.na(PRED_CLASS_APP))
    tp <- sum(df$ARR_DEL15 == 1 & df$PRED_CLASS_APP == 1, na.rm = TRUE)
    tn <- sum(df$ARR_DEL15 == 0 & df$PRED_CLASS_APP == 0, na.rm = TRUE)
    fp <- sum(df$ARR_DEL15 == 0 & df$PRED_CLASS_APP == 1, na.rm = TRUE)
    fn <- sum(df$ARR_DEL15 == 1 & df$PRED_CLASS_APP == 0, na.rm = TRUE)
    accuracy <- (tp + tn) / max(tp + tn + fp + fn, 1)
    precision <- tp / max(tp + fp, 1)
    recall <- tp / max(tp + fn, 1)
    f1 <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
    data.frame(
      Metric = c("Threshold", "Accuracy", "Precision", "Recall", "F1"),
      Value = c(input$threshold, accuracy, precision, recall, f1)
    ) %>% mutate(Value = round(Value, 4))
  })

  predict_probability <- reactive({
    carrier_high <- input$predCarrier %in% c("B6", "F9", "G4", "OH", "OO", "AA")
    airport_high <- input$predOrigin %in% c("EWR", "LGA", "ORD", "SFO")
    evening <- input$predHour >= 15 && input$predHour <= 21
    weekend <- as.integer(input$predDow) %in% c(6, 7)
    long_distance <- input$predDistance > 1500
    logit <- -2.7 +
      0.055 * input$predDepDelay +
      ifelse(evening, 0.55, 0) +
      ifelse(weekend, 0.15, 0) +
      ifelse(carrier_high, 0.30, 0) +
      ifelse(airport_high, 0.30, 0) +
      ifelse(long_distance, -0.10, 0)
    pmin(pmax(1 / (1 + exp(-logit)), 0.01), 0.99)
  })

  output$riskLabel <- renderText({
    p <- predict_probability()
    cls <- ifelse(p >= input$predThreshold, "Predicted: Delayed", "Predicted: Not Delayed")
    risk <- ifelse(p < 0.25, "Low Risk", ifelse(p < 0.60, "Medium Risk", "High Risk"))
    paste(cls, " | ", risk)
  })

  output$riskProbability <- renderText({
    paste0("Estimated probability of >15 minute arrival delay: ", percent(predict_probability(), accuracy = 0.1))
  })

  output$riskGauge <- renderPlot({
    p <- predict_probability()
    df <- tibble(x = "Risk", y = p)
    ggplot(df, aes(x = x, y = y)) +
      geom_col(width = 0.45) +
      coord_flip() +
      scale_y_continuous(labels = percent, limits = c(0, 1)) +
      labs(x = NULL, y = "Predicted Probability") +
      theme_minimal(base_size = 14) +
      theme(axis.text.y = element_blank(), panel.grid.major.y = element_blank())
  })

  output$riskExplanation <- renderText({
    p <- predict_probability()
    reasons <- c()
    if (input$predDepDelay > 15) reasons <- c(reasons, "the flight already has a meaningful departure delay")
    if (input$predHour >= 15 && input$predHour <= 21) reasons <- c(reasons, "the scheduled departure is during a higher-risk afternoon or evening window")
    if (input$predCarrier %in% c("B6", "F9", "G4", "OH", "OO", "AA")) reasons <- c(reasons, "the selected carrier has a higher observed delay profile in the sample dashboard logic")
    if (input$predOrigin %in% c("EWR", "LGA", "ORD", "SFO")) reasons <- c(reasons, "the selected origin airport is treated as a higher-disruption airport in the prototype")
    if (length(reasons) == 0) reasons <- c("the selected inputs do not contain strong high-risk signals in the prototype logic")
    paste0("The estimated risk is ", percent(p, accuracy = 0.1), ". This result is mainly driven by ", paste(reasons, collapse = ", "), ".")
  })
}

shinyApp(ui = ui, server = server)
