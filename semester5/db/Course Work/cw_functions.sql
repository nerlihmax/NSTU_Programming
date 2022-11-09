create or replace function add_n_products(n integer) returns char
as
$$
declare
    t int;
begin
    select max(id) into t from products;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into products (name, date_of_issue, country_id, date_of_supply, is_defect, provider_id,
                                  price)
            values ((select (array ['Балда', 'Вентерь','Удочка', 'Гарпун', 'Грузило', 'Катушка', 'Леска', 'Крючок', 'Спиннинг', 'Закидушка', 'Поплавок'])[floor(random() * 11 + 1)]),
                    (select timestamp '2010-01-10 20:00:00' +
                            random() * (timestamp '2019-01-20 20:00:00' -
                                        timestamp '2010-01-10 20:00:00')),
                    (select floor(random() * 12 + 1)),
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')),
                    (select random() > 0.97),
                    (select floor(random() * 20 + 1)),
                    (select floor(random() * 100000 + 200)));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

create or replace function add_n_sales(n integer) returns char
as
$$
declare
    t int;
begin
    select max(id) into t from sales;
    if t is null then
        select 0 into t;
    end if;
    for _ in (t + 1)..(n + t + 1)
        loop
            insert into sales (product_id, quantity, date_of_sale)
            values ((select id from products where id > floor(random() * 10000 + 1) and is_defect = false limit 1),
                    (select floor(random() * 10 + 1)),
                    (select timestamp '2020-01-10 20:00:00' +
                            random() * (timestamp '2022-08-20 20:00:00' -
                                        timestamp '2020-01-10 20:00:00')));
        end loop;
    return 'Inserted ' || n || ' elements';
end;
$$ language 'plpgsql';

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
    return query select product_info.*
                 from product_info
                 where product_info.price between a and b;
end;
$$ language plpgsql;

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
    return query select product_info.*
                 from product_info
                          inner join providers p on product_info.provider = p.name
                 where p.id = provider_id;
end;
$$ language plpgsql;

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
    from product_info
    where date_of_sale is null;

    select count(name)
    into cheapest
    from product_info
    where product_info.price < _price
      and date_of_sale is null;

    return round((cheapest::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

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
    return query select product_info.*
                 from product_info
                 where product_info.date_of_issue = _date;
end;
$$ language plpgsql;

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
    return query select product_info.*
                 from product_info
                 where _date @> product_info.date_of_sale;
end;
$$ language plpgsql;

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

create or replace function products_with_price_higher_than_given_providers_avg(_provider_id integer)
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
    return query select product_info.*
                 from product_info
                          inner join providers provider on product_info.provider = provider.name
                 where product_info.price > (select avg(product_info.price)
                                             from product_info
                                                      inner join providers p on product_info.provider = p.name
                                             where p.id = _provider_id);
end;
$$ language plpgsql;