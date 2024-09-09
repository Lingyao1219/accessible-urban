pacman::p_load(brms, data.table, car, mgcv, dplyr, mgcViz, spdep, sf, ggplot2, plyr, pls, psych, mdatools, broom, pals, metR)

# Read data
park_df <- read.csv('D://Google_Review//Accessible//bg_senti_access.csv')
park_df$CBSA <- as.factor(park_df$CBSA)
# park_df$Disability <- park_df$Disability_below_poverty_R+park_df$Disability_above_poverty_R
# summary(park_df)

# Scale the data
ind <- sapply(park_df, is.numeric)
ind[(names(ind) %in% "weight_st")] <- FALSE
park_df[ind] <- scale(park_df[ind], center = TRUE, scale = TRUE)
park_df[is.na(park_df)] <- 0
ggplot(park_df, aes(x=weight_st)) + geom_histogram(binwidth=.1, colour="black", fill="white")

# VIF, Linear Regression Assumptions and Diagnostics
# colnames(park_df)
linear_test <- lm(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
        + Population_Density + Employment_Density + Rural_Population_R
        + Household_Below_Poverty_R + Education_Degree_R
        + Male_R + Over_65_R + Bt_45_64_R + Disability
        + Accessible_review_density + avg_poi_score, data = park_df)
vif(linear_test)
summary(linear_test, robust = TRUE)

# # Drop outliers
model.diag.metrics <- augment(linear_test)
park_df$.cooksd  <- model.diag.metrics$.cooksd
influential <- as.numeric(row.names(park_df)[(park_df$.cooksd > (10 / (nrow(park_df)-10)))]) # default: 2
print(length(influential))
park_df1 <- park_df[!(row.names(park_df) %in% influential),]
# for (var in c('Asian_R', 'Black_Non_Hispanic_R', 'HISPANIC_LATINO_R', 'Population_Density', 'Employment_Density',
#               'Household_Below_Poverty_R', 'Transit_Freq_Area', 'Zero_car_R', 'One_car_R', 'Parking_poi_density',
#               'Male_R', 'Over_65_R', 'Bt_45_64_R', 'avg_poi_score')){
#   nrow_raw <- nrow(park_df1)
#   park_df1 <- park_df1[park_df1[,var]<quantile(park_df1[,var],0.999), ]
#   print(nrow(park_df1) - nrow_raw)}

GAM_RES1 <- mgcv::bam(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
        + Population_Density + Employment_Density + Rural_Population_R
        + Household_Below_Poverty_R + Education_Degree_R
        + Male_R + Over_65_R + Bt_45_64_R + Disability
        + Accessible_review_density + avg_poi_score
        + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'), data = park_df1,
        control = gam.control(trace = TRUE), method = "fREML", discrete = TRUE)
summary(GAM_RES1, robust = TRUE)

yvar <- 'access_bg'
new.summary <- summary(GAM_RES1, robust = TRUE)
Gamm_stable <- as.data.frame(new.summary$s.table)
Gam_summary_coeff <- as.data.frame(new.summary$p.table)
names(Gamm_stable) <- names(Gam_summary_coeff)
Gam_summary_coeff <- rbind(Gam_summary_coeff, Gamm_stable)
Gam_summary_coeff <- data.frame(names = row.names(Gam_summary_coeff), Gam_summary_coeff)
Gam_summary_coeff$Yvar <- yvar
Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("Adj_R2",new.summary$r.sq,'','','',yvar)
Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("dev.expl",new.summary$dev.expl,'','','',yvar)
Gam_summary_coeff[nrow(Gam_summary_coeff) + 1,] <- c("n",new.summary$n,'','','',yvar)
fwrite(Gam_summary_coeff, paste('D:\\Google_Review\\Accessible\\', yvar, '_Gam_scale.csv'))

# # Nonlinear: Interaction by category
# GAM_RES1 <- mgcv::bam(weight_st ~ Asian_R + Black_Non_Hispanic_R + HISPANIC_LATINO_R + Indian_R
#             + Employment_Density  + Rural_Population_R + Population_Density
#             + Walkability + Transit_Freq_Area + Zero_car_R + One_car_R + Parking_poi_density
#             + Democrat_R + Male_R + Over_65_R + Bt_45_64_R + avg_poi_score + Categories
#             + s(Education_Degree_R, Household_Below_Poverty_R, by = Categories) #+ Categories
#             # + s(Household_Below_Poverty_R, by = Categories)
#             + ti(Lat, Lng, bs = 'gp') + s(CBSA, bs = 're'), data = park_df1,
#         control = gam.control(trace = TRUE), method = "fREML", discrete = TRUE)
# summary(GAM_RES1)
# b <- getViz(GAM_RES1, scheme = 3)
# pl <- plot(b, select = 2:7) # , select = 2:7
# print(pl, pages = 1)