create or replace view product_info as
select products.name::text     as name,
       date_of_issue::date     as date_of_issue,
       provider.name::text     as provider,
       price::integer          as price,
       country.name::text      as country,
       sale.date_of_sale::date as date_of_sale
from products
         inner join providers provider on products.provider_id = provider.id
         inner join countries country on products.country_id = country.id
         inner join sales sale on products.id = sale.product_id;

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

--============

--Prices diversion
select min(price) || ' RUB'        as min_price,
       round(avg(price)) || ' RUB' as avg_price,
       max(price) || ' RUB'        as max_price
from product_info;

create or replace function products_price_between(a integer, b integer)
    returns table
            (
                name          text,
                date_of_issue date,
                provider      text,
                price         integer,
                country       text,
                date_of_sale  date
            )
as
$$
declare
    _a integer;
    _b integer;
begin
    select a into _a;
    select b into _b;
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale
                 from product_info
                 where product_info.price between a and b;
end;
$$ language 'plpgsql';

--Price between 100 and 10000
select *
from products_price_between(100, 10000);

create or replace function products_specific_provider(provider_id integer)
    returns table
            (
                name          text,
                date_of_issue date,
                provider      text,
                price         integer,
                country       text,
                date_of_sale  date
            )
as
$$
begin
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale
                 from product_info
                          inner join providers p on product_info.provider = p.name
                 where p.id = provider_id;
end;
$$ language 'plpgsql';

--Products with specified provider
select *
from products_specific_provider(5);