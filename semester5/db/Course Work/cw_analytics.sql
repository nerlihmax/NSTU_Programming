--Product info
select *
from product_info;

--Product info by date
select *
from product_info
order by date_of_issue;

--Product info by provider
select *
from product_info
order by provider;

--Prices diversion
select min(price) || ' RUB'        as min_price,
       round(avg(price)) || ' RUB' as avg_price,
       max(price) || ' RUB'        as max_price
from product_info;

--Price between 100 and 10000
select *
from products_price_between(100, 10000);

--Products with specified provider
select *
from products_specific_provider(5);

--Products cheaper than 25000
select products_cheaper_than(25000) || '%' as products_cheaper_than;

--Products with specified date
select *
from products_with_specific_date('2015-10-02'::date);

--Products with date interval
select *
from products_with_date_interval('[2021-01-01, 2021-12-31]'::daterange);

--Products sold during period
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange) || '%' as percentage;

--Products sold during period with provider
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange, 1) || '%' as percentage;


--Defected products from country with provider
select defected_products_from_country_and_provider(10, 1);

--Products by provider more expensive than average price for products from country
select *
from products_by_provider_more_expensive_than_avg(1, 1);

--Products cheaper than specified with specified provider
select products_cheaper_than_specified(10000, 4) || '%' as percentage;

