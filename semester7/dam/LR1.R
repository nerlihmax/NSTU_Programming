table <- read.table(
    "~/code/NSTU_LABS/semester7/dam/v8.csv", 
    sep = ';', 
    header = TRUE,
    col.names = c("index", "group", "gender", "age", "purchases_year", "average_price_year", "average_pages_visit", "tickets_year", "polls", "services_rating_score", "services_rating_quality")
)
table$gender <- as.factor(table$gender)
table$group <- as.factor(table$group)
table$polls <- as.factor(table$polls)
table$services_rating_quality <- as.factor(table$services_rating_quality)

data1 <- subset(table, group == 1)
data2 <- subset(table, group == 2)

summary(data1)
summary(data2)

get_stats <- function(set) {
    if (is.numeric(set)) {
        slength <- length(set)
        smean   <- mean(set)
        sstdv   <- sd(set)
        sskew   <- sum((set - smean) ^ 3/ sstdv ^ 3) / slength    
        skurt   <- sum((set - smean) ^ 4 / sstdv ^ 4 ) / slength - 3
        
        return (c(
            length = slength,
            mean   = smean,
            stdv   = sstdv,
            skew   = sskew,
            kurt   = skurt
        ))
    }
}

get_stats(data1$age)