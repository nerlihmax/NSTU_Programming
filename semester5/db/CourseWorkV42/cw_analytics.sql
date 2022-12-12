drop view if exists photo_goods_info;
create or replace view photo_goods_info as
select goods.name                as name,                -- наименование
       goods.date_of_issue,                              -- дата производства
       provider.name             as provider,            -- поставщик
       goods.date_of_supply,                             -- дата поставки
       goods.price,                                      -- цена
       goods.weight,                                     -- вес
       goods.date_of_sale,                               -- дата продажи
       manufacturer.name         as manufacturer,        -- производитель
       manufacturer_country.name as manufacturer_country -- страна производителя
from photo_goods as goods
         inner join providers provider on provider.id = goods.provider
         inner join manufacturers manufacturer on manufacturer.id = goods.manufacturer
         inner join countries manufacturer_country on manufacturer_country.id = manufacturer.country;

-- Для каждого вида фототоваров указать сведения о нем
select *
from photo_goods_info;

-- Для каждого вида фототоваров выдать список, отсортированный по дате выпуска
select *
from photo_goods_info
order by date_of_issue;

-- Для каждого вида фототоваров выдать список, отсортированный по поставщику
select *
from photo_goods_info
order by provider;

-- Для каждого вида фототоваров выдать список, отсортированный по стоимости
select *
from photo_goods_info
order by price;

-- Найти самый дорогой вид фототоваров
select name,
       max(price) || ' RUB' as max_price
from photo_goods
group by name
order by max_price
limit 1;

-- Найти самый дешевый вид фототоваров
select name,
       min(price) || ' RUB' as min_price
from photo_goods
group by name
order by min_price
limit 1;

-- Найти среднюю стоимость по каждому виду
select name,
       round(avg(price)) || ' RUB' as avg_price
from photo_goods
group by name
order by avg_price;

-- Найти среднюю стоимость фототоваров в целом
select round(avg(price)) || ' RUB' as avg_price
from photo_goods;

-- Найти фототовары с ценой больше заданной
select *
from photo_goods_info
where price > 5000 -- ввод цены
order by price;

-- Найти долю фототоваров заданного производителя от общего числа изделий
create or replace function ratio_of_goods_of_given_manufacturer(manufacturer_name text)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info;

    select count(name)
    into target
    from photo_goods_info
    where manufacturer = manufacturer_name;
    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_goods_of_given_manufacturer('Fujifilm') || '%' as ratio;

-- Найти все фототовары с заданной датой выпуска
select *
from photo_goods_info
where date_of_issue = '2010-06-26';
-- ввод цены

-- Найти все фототовары, чья дата продажи находится в заданных пределах(ввод интервала) для заданного производителя (выбор)
select *
from photo_goods_info
where '[2020-01-01, 2020-12-31]'::daterange @> date_of_sale -- ввод интервала дат
  and provider = 'ALPA';
-- ввод поставщика

-- Найти все фототовары, чья дата продажи находится в заданных пределах(ввод интервала) в целом
select *
from photo_goods_info
where '[2020-01-01, 2020-12-31]'::daterange @> date_of_sale; -- ввод интервала дат

-- Найти долю фототоваров, чья стоимость находится в заданных пределах от общего количества фотоаппаратов
create or replace function ratio_of_goods_with_price_in_interval(min_price int, max_price int)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info;

    select count(name)
    into target
    from photo_goods_info
    where price between min_price and max_price;
    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_goods_with_price_in_interval(3000, 10000) || '%' as ratio;

-- Найти долю фототоваров, поступивших от заданного поставщика
create or replace function ratio_of_goods_of_given_provider(provider_name text)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info;

    select count(name)
    into target
    from photo_goods_info
    where provider = provider_name;
    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_goods_of_given_provider('Fujifilm') || '%' as ratio;

-- Найти все фототовары, поступившие от заданного поставщика, чья стоимость больше заданной
select *
from photo_goods_info
where provider = 'Fujifilm' -- ввод поставщика
  and price > 5000;
-- ввод цены

-- Найти все фототовары заданного производителя
select *
from photo_goods_info
where manufacturer = 'Zeiss'; -- ввод производителя

-- Найти долю фототоваров, проданных за определенный период от общего времени продажи
create or replace function ratio_of_goods_sold_by_date_range(date_of_sale_range daterange)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info
    where date_of_sale is not null;

    select count(name)
    into target
    from photo_goods_info
    where date_of_sale_range @> date_of_sale;
    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_goods_sold_by_date_range('[2022-01-01, 2022-12-31]') || '%' as ratio;

-- Найти все фототовары, поступившие от заданного поставщика, чья стоимость больше, чем средняя стоимость товаров, поступивших из заданной страны
with avg_price as (select round(avg(price))
                   from photo_goods
                            inner join providers p on p.id = photo_goods.provider
                            inner join countries c on c.id = p.country
                   where c.name = 'США') -- ввод страны
select photo_goods_info.*,
       (select * from avg_price) as avg_price
from photo_goods_info
where price > (select * from avg_price)
  and provider = 'Casio' -- ввод поставщика
order by price;

-- Найти долю дорогих товаров, чья стоимость больше заданной, поступивших за определенный период
create or replace function ratio_of_goods_with_price_gt_given(price_limit int, date_of_supply_range daterange)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info;

    select count(name)
    into target
    from photo_goods_info
    where date_of_supply_range @> date_of_supply
      and price > price_limit;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

create or replace function ratio_of_goods_with_price_gt_given(price_limit int)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from photo_goods_info;

    select count(name)
    into target
    from photo_goods_info
    where price > price_limit;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

-- Найти долю дорогих товаров, чья стоимость больше заданной в целом
select ratio_of_goods_with_price_gt_given(10000) || '%' as ratio;

-- Найти долю дорогих товаров, чья стоимость больше заданной, поступивших за определенный период
select ratio_of_goods_with_price_gt_given(10000, '[2015-01-01, 2015-12-31]') || '%' as ratio;

-- Найти среднюю стоимость фототоваров, проданных за определенный промежуток времени
select round(avg(price)) || ' RUB' as avg_price
from photo_goods_info
where '[2020-01-01, 2020-12-31]'::daterange @> date_of_sale; -- ввод интервала

-- Найти все фототовары, чья стоимость выше, чем средняя стоимость фототоваров заданного производителя
with avg_price as (select round(avg(price)) as avg_price
                   from photo_goods_info
                   where manufacturer = 'Zeiss') -- ввод производителя
select photo_goods_info.*,
       (select * from avg_price)
from photo_goods_info
where price > (select * from avg_price)
order by price;