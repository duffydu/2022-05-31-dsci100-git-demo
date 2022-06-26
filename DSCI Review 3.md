# 

---

## 改大小写

```r
title<-tolower("ABC")
title<-toupper("abc")
```

## 极值

```r
max<-max(1,2,3)
min<-min(1,2,3)
```

## Package

```r
library(tidyverse)
library(rvest)
library(dbplyr)
library(RSQLite)
```

## 读入表格

```r
table0<-read_delim("data/list0.csv",delim=";") #delim是笼统形式
table1<-read_csv("data/list1.csv") #用，分割
table2<-read_csv2("data/list2.csv") #用；分割
table3<-read_tsv("data/list3.tsv") #用tab分割
table4<-read_excel("data/list4.xlsx") #excel
table5<-read_csv("data/list5.csv",skip=2) #跳过2行
table6<-read_csv("data/list6_with_no_header",col_names=c("Year","Count","Month")) #设定表格每列的数据名称
db1<-dbConnect(RSQLite::SQLite(), 'data/flights_filtered.db') #读取database
table7<-tbl(db1,table_name) #从database里直接读取表格，table name从dbListTables()得知
```

## Database操作

```r
dbListTables(db1) #查看database包含的表格
```

## ## 计算行数

```r
number<-nrow(table1)
```

## 赋值

```r
 population <- c(39.512, 4.217, 7.615)
```

## 计算

### 平均值

```r
num_mean<-mean(num)
```

### 中位数

```r
num_median<-median(num)
```

### 标准差

```r
num_standard_deviation<-sd(num)
```

## filter #筛选行的一种方式

```r
marathon_filtered<-filter(marathon_small,sex=="male")
```

## select #选择列的一种方式

```r
marathon_male<-select(marathon_filtered,bmi,km10_time_seconds)
```

## mutate #增加列的一种方式

```r
marathon_minutes<-mutate(marathon_male,km10_time_minutes=km10_time_seconds/60)
```

## addrow() #增加一行observation

```r
newData <- small_sample %>%
    add_row(Symmetry = 0.5, Radius = 0, Class = "unknown")
```

## group_by #按照一个col里的不同类型分类统计

## summarize #统计同一类型，增加新的col

```r
avocado_aggregate <- select(avocado,average_price,wk) %>% 
    group_by(wk) %>% 
    summarize(average_price = mean(average_price, na.rm = TRUE)) #na.rm：将数据里的na转化为0
```

## pivot_longer #将表格竖着写

```r
tidy_temp <- sea_surface %>%
    pivot_longer(cols = Jan:Dec, #所选原表格需要转化的col
                 names_to = "Month",  #把他放在新的表格的col的名字
                 values_to = "Temperature") #把它的值命名为
```

## mad_df #给每一列进行操作指令

```r
pollution_2006<- madrid %>%
     filter(year == 2006) %>%
     select(-date, -year, -mnth) %>%
     map_df(max, na.rm  = TRUE) #给每一列求最大值
```

## arrange #列内排序

```r
max_pollution_diff <- pollution_diff %>% arrange(desc(value)) %>% #给value列按降序排列
    tail(n = 1) #选取末尾一行
```

## is_na #是否式NA

```r
world_vaccination <- read_csv("data/world_vaccination.csv") %>% filter(!is.na(pct_vaccinated), who_region != "(WHO) Global") #找出cyl不为NA的行
```

## n() #算出当前组的大小

```r
top_restaurants <- fast_food %>%
    filter(st %in% c("CA", "WA", "OR")) %>% 
    group_by(name) %>%
    summarize(n = n()) %>% #按照name分类，计算出分别name的组的大小
    arrange(desc(n)) %>% #降序排列
    head(n = 9) #显示前9组
```

## as. #改变显示数据类型

```r
cancer <- cancer %>%
        mutate(Class = as_factor(Class)) #将数据改成类型类
point_a <- slice(cancer, 1) %>%
    select(Symmetry, Radius, Concavity) %>%
    as.numeric()                         #将数据改成数字类
dist_matrix <- newData %>%
    select(Symmetry, Radius) %>% 
    dist() %>%                   
    as.matrix()                          #将数据改成矩阵类
```

## dist() #计算两点距离

```r
dist_cancer_two_rows <- cancer  %>% 
    slice(1,2)  %>% 
    select(Symmetry, Radius, Concavity)  %>% 
    dist()
```

## sample_n() #随机选取单个样本

```r
small_sample <- sample_n(cancer, 5)
```

## 分布

### rep_sample_n() #随机选取n个含有k个观测值的样本

```r
rep_sample_n(can_senior , 40) #同sample_n

samples <-rep_sample_n(can_seniors, size = 1500, reps = 40) 
#选取1500个样本，每个样本含有40条观测值

samples<-rep_sample_n(can_seniors, size = 1500, replace=TRUE,reps = 40)
```

### 基本操作

```r
#第一组
head(samples)
#最后一组
tail(samples)
#数据的维度
dim(samples)
```

### 计算每组样本的平均值

```r
sample_estimates<-samples%>%
                group_by(replicate)%>%
                summarize(sample_mean=mean(age))

#画图
sampling_distribution<-sample_estimates%>%
                        ggplot(aes(x=sample_mean))+
                        geom_histogram()+
                        labs(x="mean")+
                        ggtitle("Mean Distribution")
```

### 计算置信区间

```r
bounds <- boot20000_means |>
  select(mean) |>
  pull() |>
  quantile(c(0.025, 0.975)) #95置信区间
```

## ggplot #绘图

### 图片大小设置

```r
options(repr.plot.width = 8, repr.plot.height = 7)
```

### 散点图

```r
ggplot(data = marathon_minutes, aes(x = bmi, y = km10_time_minutes)) + 
    geom_point() + 
    theme(text = element_text(size=20)) #修改字体大小

#添加颜色和形状属性
compare_vacc_plot<-ggplot(world_vaccination,aes(x=yr,y=pct_vaccinated))+
    geom_point(aes(colour = vaccine, shape = vaccine))+ #颜色和形状数据来源
    labs(x="Year",y="People Vaccinated",colour="Vaccine",shape="Vaccine")
```

### 直方图

```r
ggplot(faithful, aes(x = waiting)) + 
    geom_histogram(bins = 40) + #x轴上的多少小格
    xlab("Waiting Time (mins)") + 
    ylab("Count") + 
    theme(text = element_text(size=20))


arrival_delay_plot <- ggplot(delay_data, aes(x = ARRIVAL_DELAY/60)) +
   geom_histogram(aes(y = 100 * stat(count) / sum(stat(count))), #更改y轴count的scale
        binwidth = .25,  #更改x轴scale
        fill = "lightblue",  #直方的填充颜色
        color = 'steelblue') + #直方的边框颜色
    scale_x_continuous(limits = c(-2, 5)) + #规定x的范围
    ylab("% of Flights") +
    xlab("Delay (hours)") +
    theme(text = element_text(size=20))
```

### 折线图

```r
polio_regions_line <- ggplot(polio,aes(x=yr,y=pct_vaccinated))+
    geom_line(aes(colour=who_region))+
    labs(x="Year",y="percentage vaccinated",colour="Region")
```

### 柱形图

```r
count_bar_chart <- ggplot(top_restaurants, aes(x = name, y = n)) + 
    geom_bar(stat="identity") + #按照原样显示
    xlab("Restaurant") +
    ylab("Count")

#调整柱形图的x轴字体的倾斜程度
count_bar_chart_A <- 
    count_bar_chart+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

#xy轴对换
count_bar_chart_B <- count_bar_chart + 
    coord_flip()

#fill 填色
bar_plot <- insurance %>%
    ggplot(aes(x = sex, fill = smoker)) + 
    geom_bar(position = 'fill') + 
    xlab("Sex") +
    ylab("Count") +
    labs(fill = "Does the person smoke") +
    ggtitle("Smokers in different sex")

#修改填色
library(RColorBrewer)
display.brewer.all()
diamonds_plot <- diamonds_plot + 
       scale_fill_brewer(palette = 1)
```

### facet_grid #将两个图表并排放或竖着放

```r
side_by_side_world <- ggplot(world_vaccination,aes(x=yr,y=pct_vaccinated)) +
    geom_line(aes(colour=who_region)) +
    labs(colour="Region") +
    xlab("Year") + 
    ylab("percentage vaccinated") +
    facet_grid(col=vars(vaccine)) #按照vaccine的数据分类，横着放两个表格，如果是竖着放就是row=... 

vertical_world<-ggplot(world_vaccination,aes(x=yr,y=pct_vaccinated)) +
    geom_line(aes(colour=who_region)) +
    labs(colour="Region") +
    xlab("Year") + 
    ylab("percentage vaccinated") +
    facet_grid(row=vars(vaccine))
```

### plot_grid #将n个图表并排放在一起

```r
plot_gird(p1,p2,p3,ncol=3)
```

### facet_wrap #将所有图表给打包放在一起

```r
all_temp_plot <- tidy_temp %>% 
    ggplot(aes(x = Year, y = Temperature)) + 
    geom_point() + 
    facet_wrap(~ factor(Month, levels = c("Jan","Feb","Mar","Apr","May","Jun",
                                          "Jul","Aug","Sep","Oct","Nov","Dec"))        ) +
    #原表格有列为month对应所有月份的
    xlab("year") + 
    ylab("temperature") +
    theme(text = element_text(size=20))
```

### lim() #设置x，y轴范围

```r
sampling_distribution_20<-sample_estimates%>%
                        ggplot(aes(x=sample_mean))+
                        geom_histogram()+
                        labs(x="mean")+
                        ggtitle("Mean Distribution")+
                        xlim(c(65,95)) #设置x轴的范围
```

## 建模

### Nearest Neighbor Classification

```r
fruit_split <- initial_split(fruit_data, prop = 3/4, strata = fruit_name) #按75%比例分割数据 
fruit_train <- training(fruit_split)   #设置训练数据
fruit_test  <- testing(fruit_split)    #设置测试数据

fruit_recipe <- recipe(fruit_name ~ mass+color_score , data = fruit_train) %>%
step_scale(all_predictors()) %>%
step_center(all_predictors())         


knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) %>%
       set_engine("kknn") %>%
       set_mode("classification")

fruit_fit <- workflow() %>%
       add_recipe(fruit_recipe) %>%
       add_model(knn_spec) %>%
       fit(data = fruit_train)

fruit_test_predictions<- predict(fruit_fit,fruit_test) %>%
       bind_cols(fruit_test) #将预测的测试数据与测试数据结合成一个表格


fruit_prediction_accuracy <- fruit_test_predictions %>%
         metrics(truth = fruit_name, estimate = .pred_class) 
 #生成含有预测准确性的表格

fruit_mat <- fruit_test_predictions %>% 
        conf_mat(truth = fruit_name, estimate = .pred_class)  
 #查看有哪些预测出现问题的矩阵

#Cross-validation
fruit_vfold <- vfold_cv(fruit_train, v = 5, strata = fruit_name)

fruit_resample_fit <- workflow() %>%
       add_recipe(fruit_recipe) %>%
       add_model(knn_spec) %>%
       fit_resamples(resamples = fruit_vfold)
#fit这一步稍微不太一样，其他都一样

fruit_metrics<-collect_metrics(fruit_resample_fit)
#生成含有平均准确率及标准差的表格

#tune()
knn_tune<-nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
       set_engine("kknn") %>%
       set_mode("classification")

knn_results <- workflow() %>%
       add_recipe(fruit_recipe) %>%
       add_model(knn_tune) %>%
       tune_grid(resamples = fruit_vfold, grid = 10) %>%
       collect_metrics()   
#你会得到一个表格，上面列举了不同k值的精确度

#可以指定tune的范围
gridvals<-tibble(neighbors=1:200) #指定k从1试到200
marathon_results<-
       tune_grid(marathon_workflow,resamples = marathon_vfold, grid = gridvals) %>%
                       collect_metrics() 
# 或者
ks <- tibble(neighbors = seq(from = 1, to = 10, by = 1)) #by说明每个数字的间隔
```

### Nearest Neighbor Regression

```r
marathon_recipe <- recipe(time_hrs ~ max, data = marathon_training) %>%
       step_scale(all_predictors()) %>%
       step_center(all_predictors())

#tune后的best k
marathon_best_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = k_min) %>%
          set_engine("kknn") %>%
          set_mode("regression")

marathon_best_fit <- workflow() %>%
          add_recipe(marathon_recipe) %>%
          add_model(marathon_best_spec) %>%
          fit(data = marathon_training)

marathon_summary <- marathon_best_fit %>%
           predict(marathon_testing) %>%
           bind_cols(marathon_testing)%>%
           metrics(truth = time_hrs, estimate = .pred) 
```

### Linear Regression

```r
lm_spec<-linear_reg()%>%
        set_engine("lm")%>%
        set_mode("regression")

lm_recipe <- recipe(time_hrs ~ max, data = marathon_training)

lm_fit <- workflow() %>%
       add_recipe(lm_recipe) %>%
       add_model(lm_spec) %>%
       fit(marathon_training)
#画含有线性回归的图
lm_predictions<-ggplot(marathon_training,aes(x=max,y=time_hrs))+
            geom_point()+
            geom_smooth(method = "lm", se = FALSE)+
                labs(x="Time in hours",y="Max training distance")
```

### Clustering

#### Scale数据

```r
scaled_beer <- clean_beer %>% 
    mutate(across(everything(), scale))
```

#### Setting center for clustering

```r
beer_cluster_k2 <- kmeans(scaled_beer, centers = 2)
```

#### 用augment将数据放进每个clustering里

```r
tidy_beer_cluster_k2 <- augment(beer_cluster_k2, scaled_beer)
```

#### 用glance获取model-level statistics

```r
beer_cluster_k2_model_stats<-glance(beer_cluster_k2)
```

#### 找到最好的k值

```r
#创建1-10的表格
beer_ks<-tibble(k=1:10)

beer_clustering <- beer_ks %>%
     rowwise() %>%
     mutate(models = list(kmeans(scaled_beer, center=2)))


beer_model_stats <- beer_clustering %>% 
     mutate(model_statistics = list(glance(models,k)))

#会得到一个1-10各自的model-level statistics
#绘制图像找到最佳值
options(repr.plot.width = 8, repr.plot.height = 7)
choose_beer_k<-ggplot(beer_clustering_unnested,aes(x=k,y=tot.withinss))+
                geom_line()+
                geom_point()+
labs(x="K",y="beer_clustering_unnested")
```
