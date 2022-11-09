create or replace view product_info as
select products.name::text     as name,
       date_of_issue::date     as date_of_issue,
       provider.name::text     as provider,
       price::integer          as price,
       country.name::text      as country,
       sale.date_of_sale::date as date_of_sale,
       is_defect::boolean      as is_defect
from products
         inner join providers provider on products.provider_id = provider.id
         inner join countries country on products.country_id = country.id
         left join sales sale on products.id = sale.product_id;

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
                date_of_sale  date,
                is_defect     boolean
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
                        product_info.date_of_sale,
                        product_info.is_defect
                 from product_info
                 where product_info.price between a and b;
end;
$$ language plpgsql;

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
                date_of_sale  date,
                is_defect     boolean
            )
as
$$
begin
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale,
                        product_info.is_defect
                 from product_info
                          inner join providers p on product_info.provider = p.name
                 where p.id = provider_id;
end;
$$ language plpgsql;

--Products with specified provider
select *
from products_specific_provider(5);

create or replace function products_cheaper_than(_price integer)
    returns integer
as
$$
declare
    cheapest     integer;
    all_products integer;
begin
    select count(name)
    into all_products
    from product_info;

    select count(name)
    into cheapest
    from product_info
    where product_info.price < _price;

    return round((cheapest::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

--Products cheaper than 25000
select products_cheaper_than(25000) || '%' as products_cheaper_than;

create or replace function products_with_specific_date(_date date)
    returns table
            (
                name          text,
                date_of_issue date,
                provider      text,
                price         integer,
                country       text,
                date_of_sale  date,
                is_defect     boolean
            )
as
$$
begin
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale,
                        product_info.is_defect
                 from product_info
                 where product_info.date_of_issue = _date;
end;
$$ language plpgsql;

--Products with specified date
select *
from products_with_specific_date('2015-10-02'::date);

create or replace function products_with_date_interval(_date daterange)
    returns table
            (
                name          text,
                date_of_issue date,
                provider      text,
                price         integer,
                country       text,
                date_of_sale  date,
                is_defect     boolean
            )
as
$$
begin
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale,
                        product_info.is_defect
                 from product_info
                 where _date @> product_info.date_of_sale;
end;
$$ language plpgsql;

--Products with date interval
select *
from products_with_date_interval('[2021-01-01, 2021-12-31]'::daterange);

create or replace function products_sold_during_period(_date daterange)
    returns integer
as
$$
declare
    sold_during_period integer;
    all_products       integer;
begin
    select count(name)
    into all_products
    from product_info
    where date_of_sale is not null;

    select count(name)
    into sold_during_period
    from product_info
    where _date @> product_info.date_of_sale
      and date_of_sale is not null;

    return round((sold_during_period::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;
create or replace function products_sold_during_period(_date daterange, _provider_id integer)
    returns integer
as
$$
declare
    sold_during_period integer;
    all_products       integer;
begin
    select count(product_info.name)
    into all_products
    from product_info
             inner join providers p on product_info.provider = p.name
    where date_of_sale is not null
      and p.id = _provider_id;

    select count(product_info.name)
    into sold_during_period
    from product_info
             inner join providers p on product_info.provider = p.name
    where _date @> product_info.date_of_sale
      and date_of_sale is not null
      and p.id = _provider_id;

    return round((sold_during_period::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

--Products sold during period
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange) || '%' as percentage;

--Products sold during period with provider
select products_sold_during_period('[2021-01-01, 2021-12-31]'::daterange, 1) || '%' as percentage;

create or replace function defected_products_from_country_and_provider(_country_id integer, provider_id integer)
    returns integer
as
$$
declare
    count integer;
begin
    select count(product_info.name)
    into count
    from product_info
             inner join countries country on product_info.country = country.name
             inner join providers p on product_info.provider = p.name
    where product_info.is_defect = true
      and country.id = _country_id
      and provider_id = p.id;
    return count;
end;
$$ language plpgsql;

--Defected products from country with provider
select defected_products_from_country_and_provider(10, 1);

create or replace function products_by_provider_more_expensive_than_avg(_country_id integer, provider_id integer)
    returns table
            (
                name          text,
                date_of_issue date,
                provider      text,
                price         integer,
                country       text,
                date_of_sale  date,
                is_defect     boolean
            )
as
$$
begin
    return query select product_info.name,
                        product_info.date_of_issue,
                        product_info.provider,
                        product_info.price,
                        product_info.country,
                        product_info.date_of_sale,
                        product_info.is_defect
                 from product_info
                          inner join providers p on product_info.provider = p.name
                 where p.id = provider_id
                   and product_info.price > (select avg(product_info.price)
                                             from product_info
                                                      inner join countries c on product_info.country = c.name
                                             where c.id = _country_id);
end;
$$ language plpgsql;

--Products by provider more expensive than average price for products from country
select *
from products_by_provider_more_expensive_than_avg(1, 1);

create or replace function products_cheaper_than_specified(_price integer, _provider_id integer) returns integer
as
$$
declare
    all_products integer;
    cheapest     integer;
begin
    select count(product_info.name)
    into all_products
    from product_info
             inner join providers p on product_info.provider = p.name
    where date_of_sale is null
      and p.id = _provider_id;

    select count(product_info.name)
    into cheapest
    from product_info
             inner join providers p on product_info.provider = p.name
    where product_info.price < _price
      and date_of_sale is null
      and p.id = _provider_id;
    return round((cheapest::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

--Products cheaper than specified with specified provider
select products_cheaper_than_specified(10000, 4) || '%' as percentage;

