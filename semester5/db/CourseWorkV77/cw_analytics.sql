create or replace view equipments_info as
select equipments.id             as id,
       equipments.name,
       manufacturer.name         as manufacturer,
       manufacturer_country.name as manufacturer_country,
       equipments.date_of_issue,
       provider.name             as provider,
       equipments.date_of_supply,
       equipments.age_category,
       equipments.price
from equipments
         inner join manufacturers manufacturer on manufacturer.id = equipments.manufacturer
         inner join providers provider on provider.id = equipments.provider
         inner join countries manufacturer_country on manufacturer_country.id = manufacturer.country;

-- Для каждого вида спортивных товаров указать сведения о нём
select *
from equipments_info
order by name;

-- Для каждого вида спортивных товаров выдать список отсортированный по дате выпуска
select *
from equipments_info
order by date_of_issue;

-- Для каждого вида спортивных товаров выдать список отсортированный по поставщику
select *
from equipments_info
order by provider;

-- Для каждого вида спортивных товаров выдать список отсортированный по стоимости
select *
from equipments_info
order by price;

-- Найти самый дорогой вид спортивных товаров
select name, price
from equipments_info
where price = (select max(price) from equipments_info);

-- Найти самый дешевый вид спортивных товаров
select name, price
from equipments_info
where price = (select min(price) from equipments_info);

-- Найти среднюю стоимость по каждому виду
select name, round(avg(price)) as avg_price
from equipments_info
group by name;

-- Найти среднюю стоимость спортивных товаров
select round(avg(price)) as avg_price
from equipments_info;

-- Найти спортивные товары с ценой выше 2000р. (ввод цены)
select *
from equipments_info
where price > 2000; -- ввод цены

-- Найти долю спортивных товаров заданного производителя от общего числа изделий
create or replace function ratio_of_equipment_from_given_manufacturer(_manufacturer text)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
    where manufacturer = _manufacturer;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_equipment_from_given_manufacturer('Adidas') || '%' as ratio;

-- Найти все спортивные с заданной датой выпуска
select *
from equipments_info
where date_of_issue = '2010-01-14';

-- Найти все спортивные товары, чья дата выдачи в прокат находится в заданных пределах
select equipments_info.name,
       er.start_of_lease,
       er.end_of_lease
from equipments_info
         inner join equipment_rental er on er.equipment = equipments_info.id
where '[2020-01-01, 2020-12-31]'::daterange @> start_of_lease; -- ввод пределов даты

-- Найти долю спортивных товаров, чья стоимость находится в заданных пределах
create or replace function ratio_of_equipment_which_price_in_interval(min_price int, max_price int)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
    where price between min_price and max_price;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;
select ratio_of_equipment_which_price_in_interval(2000, 5000) || '%' as ratio;

-- Найти долю спортивных товаров, поступивших от заданного поставщика
create or replace function ratio_of_equipment_from_given_provider(_provider text)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
    where provider = _provider;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;
select ratio_of_equipment_from_given_provider('Спортмастер') || '%' as ratio;

-- Найти спортивные товары, поступивших от заданного поставщика, чья стоимость больше заданного
select name, provider, price
from equipments_info
where provider = 'Спортмастер' -- ввод поставщика
  and price > 5000;
-- ввод цены

-- Найти все спортивные товары заданного производителя
select *
from equipments_info
where manufacturer = 'Adidas';

-- Найти долю спортивных товаров, выданных для проката за определённый период
create or replace function ratio_of_equipment_which_lease_began_at_interval(lease_start_interval daterange)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
             inner join equipment_rental er on er.equipment = equipments_info.id
    where lease_start_interval @> er.start_of_lease;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;
select ratio_of_equipment_which_lease_began_at_interval('[2020-01-01, 2020-12-31]') || '%' as ratio;

-- Найти все спортивные товары, поступившие от заданного поставщика, чья стоимость больше, чем средняя стоимость товаров, поступивших из заданной страны
with avg_price as (select round(avg(price)) from equipments_info where manufacturer_country = 'США') -- ввод страны
select equipments_info.*,
       (select * from avg_price) as avg_price
from equipments_info
where price > (select * from avg_price)
  and provider = 'Спортмастер' -- ввод поставщика
order by price;

-- Найти долю дорогих товаров (чья стоимость больше заданной), поступивших за определенный период и в целом
create or replace function ratio_of_expensive_equipment(price_threshold int, supply_interval daterange)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
    where supply_interval @> equipments_info.date_of_supply
      and price > price_threshold;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

create or replace function ratio_of_expensive_equipment(price_threshold int)
    returns integer
as
$$
declare
    target    integer;
    all_goods integer;
begin
    select count(name)
    into all_goods
    from equipments_info;

    select count(name)
    into target
    from equipments_info
    where price > price_threshold;

    return round((target::double precision / all_goods::double precision) * 100);
end;
$$ language plpgsql;

select ratio_of_expensive_equipment(8000) || '%' as ratio; -- дороже в целом
select ratio_of_expensive_equipment(8000, '[2012-01-01, 2012-12-31]') || '% ' as ratio;
-- дороже и поступили в заданный интервал дат

-- Найти среднюю стоимость спортивных товаров, выданных для проката за определенный промежуток времени
select round(avg(price)) as avg_price
from equipments_info eq
         inner join equipment_rental er on er.equipment = eq.id
where '[2021-01-01, 2021-12-31]'::daterange @> er.start_of_lease;
-- ввод интервала дат

-- Найти все спортивные товары, чья стоимость выше, чем средняя стоимость спортивных товаров заданного производителя
with avg_price as (select round(avg(price)) from equipments_info where manufacturer = 'Adidas')
select eq.*,
       (select * from avg_price) as avg_price
from equipments_info eq
where price > (select * from avg_price)
order by price;

-- Найти все спортивные товары, предназначенные для детей
