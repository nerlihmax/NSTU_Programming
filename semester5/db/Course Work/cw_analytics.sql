--Product info
select *
from product_info;

--Product info ordered by date
select *
from product_info
order by date_of_issue;

--Product info ordered by provider
select *
from product_info
order by provider;

--The most expensive product, the cheapest, the average cost
select min(price) || ' RUB'        as min_price,
       round(avg(price)) || ' RUB' as avg_price,
       max(price) || ' RUB'        as max_price
from product_info;

--Products with price between given values
select *
from products_price_between(100, 10000);

--Products with specified supplier
select *
from products_specific_provider(5);

--Products cheaper than given price
select products_cheaper_than(25000) || '%' as products_cheaper_than;

--Products with a given issue date
select *
from products_with_specific_date('2015-10-02'::date);

--Goods whose date of sale is within the specified interval
select *
from products_with_date_interval('[2021-01-01, 2021-12-31]'::daterange);

--The ratio of goods sold during the period from the total time of sale
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange) || '%' as percentage;

--Goods whose date of sale is within the specified interval for a given manufacturer
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange, 1) || '%' as percentage;

--The amount of defective goods received from a given country for a given supplier
select defected_products_from_country_and_provider(10, 1);

--Find all products from a given supplier whose value is greater than the average cost of goods from a given country
select *
from products_by_provider_more_expensive_than_avg(1, 1);

--Percentage of cheap goods received from a given supplier
select products_cheaper_than_specified(10000, 4) || '%' as percentage;

--Average cost of goods sold over a period of time
select round(avg(price)) || ' RUB' as avg_price
from products_with_date_interval('[2021-01-01, 2021-12-31]'::daterange);

--Goods whose price is higher than the average cost of a given producer
select * from products_with_price_higher_than_given_providers_avg(1);

-- The ratio of regular deliveries of a given product
-- select

--Products that sell best with given parameters: cost, supplier, country
--select