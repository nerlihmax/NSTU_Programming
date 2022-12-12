create or replace view plants as
select plants.name       as plant,
       plants.date_of_issue,
       manufacturer.name as manufacturer,
       plants.price,
       plants.sell_by,
       provider.name     as provider,
       plants.flowering_period,
       plants.storage_temperature,
       disease.name      as target_disease,
       c.name            as provider_city
from medicinal_plants as plants
         inner join manufacturers manufacturer on manufacturer.id = plants.manufacturer
         inner join providers provider on provider.id = plants.provider
         inner join cities c on c.id = provider.city
         inner join diseases disease on disease.id = plants.target_disease;


--Для каждого лекарственного растения указать сведения о нем
select *
from plants;

--Получить список растений, отсортированный по дате выпуска
select *
from plants
order by date_of_issue;

--Получить список растений, отсортированный по поставщику (в алф порядке)
select *
from plants
order by provider;

--Получить список растений, отсортированный по стоимости
select *
from plants
order by price;

--Найти самое дорогое лекарственное растение, самое дешевое, среднюю стоимость.
select min(price) || ' RUB'        as min_price,
       max(price) || ' RUB'        as max_price,
       round(avg(price)) || ' RUB' as avg_price
from plants;

--для каждого типа заболевания найти лекарственные растения с ценой свыше заданной (300 рублей для примера)
select target_disease,
       plant,
       price
from plants
where price > 9000 --Здесь задается цена
order by target_disease;

--Найти все лекарственные растения для заданного типа заболевания, чья стоимость попадает в заданные пределы
select *
from plants
where price between 1000 and 2000 --ввод интервала цены
  and target_disease = 'Воспаление почек'; --ввод заболевания

--Найти долю дешевых лекарственных растений для заданного изготовителя (объявление функции)
create or replace function ratio_of_plants_cheaper_than_given_for_manufacturer(_price integer, _manufacturer text)
    returns integer
as
$$
declare
    cheapest     integer;
    all_products integer;
begin
    select count(plant)
    into all_products
    from plants
    where plants.manufacturer = _manufacturer;

    select count(plant)
    into cheapest
    from plants
    where plants.price < _price
      and plants.manufacturer = _manufacturer;

    return round((cheapest::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

--Найти долю дешевых лекарственных растений для заданного изготовителя
select ratio_of_plants_cheaper_than_given_for_manufacturer(900, 'ООО «АНБ»') || '%' as ratio;

--Определить все лекарственные растения заданного изготовителя, чья стоимость находится в заданных пределах
select *
from plants
where price between 1000 and 9000 --ввод интервала цены
  and manufacturer = 'ООО «ШЕФ ТИ»';
--ввод изготовителя

--Найти все лекарственные растения, поступившие от заданного изготовителя для лечения заданного заболевания
select *
from plants
where manufacturer = 'ООО «ШЕФ ТИ»' --ввод изготовителя
  and target_disease = 'Грипп'; -- ввод заболевания

--Найти долю лекарственных растений для заданного поставщика (объявление функции)
create or replace function ratio_of_plants_with_given_provider(_provider text)
    returns integer
as
$$
declare
    cheapest     integer;
    all_products integer;
begin
    select count(plant)
    into all_products
    from plants;

    select count(plant)
    into cheapest
    from plants
    where plants.provider = _provider;

    return round((cheapest::double precision / all_products::double precision) * 100);
end;
$$ language plpgsql;

--Найти долю лекарственных растений для заданного поставщика
select ratio_of_plants_with_given_provider('Лавка здоровья') || '%' as ratio;

--Определить долю регулярных поставок заданного лекарственного растения (объявление функции)
create or replace function ratio_of_regular_supplies(product_name text) returns integer --Поставка считается регулярной, если за месяц было более одной поставки товара
as
$$
declare
    all_plants       double precision;
    month_counter    integer := 0;
    regular_supplies integer := 0;
begin
    for t in 1..12
        loop
            select into month_counter count(id)
            from medicinal_plants as plant
            where extract(month from (plant.date_of_supply)::timestamp) = t
              and plant.name = product_name;
            if month_counter > 1 then
                regular_supplies := regular_supplies + 1;
            end if;
            month_counter := 0;
        end loop;

    select count(plant.id)
    into all_plants
    from medicinal_plants as plant
    where plant.name = product_name;

    raise notice 'regular_supplies: %', regular_supplies;
    raise notice 'all_plants: %', all_plants;
    return round((regular_supplies::double precision / all_plants) * 100::double precision)::integer;
end;
$$
    language plpgsql;

--Определить долю регулярных поставок заданного лекарственного растения
select ratio_of_regular_supplies('Бедренец большой') || '%' as regular_supplies_ratio;

--Для заданного поставщика определить все лекарственные растения. чей температурных режим хранкения находится в заданных пределах
select *
from plants
where storage_temperature between 20 and 30 --ввод температурного диапазона
  and provider = 'Таёжный дворик'; -- ввод поставщика

--Найти долю заданных лекарственных растений, проданных за определенный период (объявление функции)
create or replace function ratio_of_plants_with_given_name_sold_in_interval(plant_name text, _interval daterange)
    returns integer
as
$$
declare
    target     integer;
    all_plants integer;
begin
    select count(name)
    into all_plants
    from medicinal_plants
             inner join sales s on medicinal_plants.id = s.plant
    where name = plant_name;

    select count(plant)
    into target
    from medicinal_plants
             inner join sales s2 on medicinal_plants.id = s2.plant
    where _interval @> s2.date
      and name = plant_name;

    return round((target::double precision / all_plants::double precision) * 100);
end;
$$ language plpgsql;

--Найти долю заданных лекарственных растений, проданных за определенный период
select ratio_of_plants_with_given_name_sold_in_interval('Брюква', '[2020-01-01, 2020-12-31]') || '%' as ratio;

--Найти все лекарственный растения, поступившие из заданного города, чья стоимость больше заданной
select plant.plant,
       plant.manufacturer,
       plant.price,
       plant.sell_by,
       plant.provider,
       plant.flowering_period,
       plant.storage_temperature,
       plant.target_disease,
       plant.provider_city
from plants as plant
where plant.provider_city = 'Вологда' -- ввод города поставщика
  and price > 4000; -- ввод цены

--Найти долю дорогих лекарственных растений (цена больше заданной) от общего числа лекарственных растений (объявление функции)
create or replace function ratio_of_plants_more_expensive_than_given_price(_price integer)
    returns integer
as
$$
declare
    target     integer;
    all_plants integer;
begin
    select count(plant)
    into all_plants
    from plants;

    select count(plant)
    into target
    from plants
    where price > _price;

    return round((target::double precision / all_plants::double precision) * 100);
end;
$$ language plpgsql;

--Найти долю дорогих лекарственных растений (цена больше заданной) от общего числа лекарственных растений
select ratio_of_plants_more_expensive_than_given_price(8700) || '%' as ratio;

--Найти самое популярное лекарственное растение, которое покупают чаще всего для заданного сезона (объявление функции)
create or replace function most_popular_plants_by_season(season flowering_period)
    returns table
            (
                name             text,
                date_of_issue    date,
                provider         text,
                price            integer,
                manufacturer     text,
                flowering_period flowering_period,
                disease          text,
                quantity         integer
            )
as
$$
begin
    return query select plant.name,
                        plant.date_of_issue,
                        p.name                     as provider,
                        plant.price,
                        m.name                     as manufacturer,
                        plant.flowering_period,
                        d.name,
                        count(plant.name)::integer as cnt
                 from medicinal_plants as plant
                          inner join sales s on plant.id = s.plant
                          inner join providers p on p.id = plant.provider
                          inner join manufacturers m on m.id = plant.manufacturer
                          inner join diseases d on d.id = plant.target_disease
                 where plant.flowering_period = season
                 group by plant.name, plant.date_of_issue, plant.price, plant.flowering_period,
                          plant.target_disease, p.name, m.name, d.name
                 order by cnt desc;
end;
$$ language plpgsql;

--Найти самое популярное лекарственное растение, которое покупают чаще всего для заданного сезона
select *
from most_popular_plants_by_season('Autumn')
limit 3; -- топ 3 по популярности

--Найти все лекарственный растения, у которых период цветения указан в заданном сезоне,
--чья стоимость больше, чем средняя стоимость лекарственных растений, проданных за определенный промежуток времени (объявление функции)
create or replace function plants_with_flowering_period_which_price_more_than_avg(date_interval daterange, season flowering_period)
    returns table
            (
                name             text,
                date_of_issue    date,
                provider         text,
                price            integer,
                manufacturer     text,
                flowering_period flowering_period,
                disease          text,
                avg_price        integer
            )
as
$$
begin
    return query with avg_price as (select avg(plant2.price)
                                    from medicinal_plants as plant2
                                             inner join sales s2 on plant2.id = s2.plant
                                    where date_interval @> s2.date)
                 select plant.name,
                        plant.date_of_issue,
                        p.name                                    as provider,
                        plant.price,
                        m.name                                    as manufacturer,
                        plant.flowering_period,
                        d.name,
                        round((select * from avg_price))::integer as avg_price
                 from medicinal_plants as plant
                          inner join sales s on plant.id = s.plant
                          inner join providers p on p.id = plant.provider
                          inner join manufacturers m on m.id = plant.manufacturer
                          inner join diseases d on d.id = plant.target_disease
                 where plant.price > (select * from avg_price)
                   and plant.flowering_period = season
                 group by plant.name, plant.date_of_issue, plant.price, plant.flowering_period,
                          plant.target_disease, p.name, m.name, d.name;
end;
$$
    language plpgsql;


--Найти все лекарственный растения, у которых период цветения указан в заданном сезоне, чья стоимость больше, чем средняя стоимость лекарственных растений, проданных за определенный промежуток времени
select *
from plants_with_flowering_period_which_price_more_than_avg(
        '[2021-01-21, 2021-12-31]', -- ввод интервала
        'Autumn' -- ввод сезона
    );