drop view if exists boat_info;
create or replace view boat_info as
select boats.vessel_type,
       boats.date_of_issue,
       boats.lifetime,
       boats_country.name        as country,
       factory.name              as manufacturer,
       manufacturer_country.name as manufacturer_country,
       boats.price,
       boats.max_speed,
       boats.status,
       boats.roominess
from boats
         inner join factories factory on factory.id = boats.factory
         inner join countries boats_country on boats.country = boats_country.id
         inner join countries manufacturer_country on manufacturer_country.id = factory.country;

--Для каждой яхты указать сведения о ней
select *
from boat_info;

--Получить список, отсортированный по году выпуска
select *
from boat_info
order by date_of_issue;

--Получить список, отсортированный по месту производства
select *
from boat_info
order by manufacturer;

--Получить список, отсортированный по стоимости
select *
from boat_info
order by price;

--Получить список, отсортированный по скорости
select *
from boat_info
order by max_speed;

--Найти долю старых яхт (срок эксплуатации больше заданного)
create or replace function ratio_of_older_boats(_interval interval)
    returns integer
as
$$
declare
    oldest    integer;
    all_boats integer;
begin
    select count(id)
    into all_boats
    from boats;

    select count(id)
    into oldest
    from boats
    where lifetime > _interval;

    return round((oldest::double precision / all_boats::double precision) * 100);
end;
$$ language plpgsql;

--Найти долю старых яхт (срок эксплуатации больше заданного)
select ratio_of_older_boats('1 year'::interval) || '%' as ratio;

--Найти самую дорогую яхту, самую дешевую, среднюю стоимость
select max(price) || ' RUB'        as max_price,
       round(avg(price)) || ' RUB' as avg_price,
       min(price) || ' RUB'        as min_price
from boats;

--Найти все яхты с ценой выше 10 млн.руб
select *
from boat_info
where price > 10000000;
-- ввод стоимости

--Найти яхты, поступившие из заданной страны, чья скорость больше заданной
select *
from boat_info
where country = 'США' --ввод страны
  and max_speed > 180;
-- ввод скорости

--Найти яхты, с заданным сроком эксплуатации
select *
from boat_info
where lifetime = interval '2 years 5 mons 13 days'; -- ввод срока эксплуатации

--Найти долю яхт, поступивших из заданной страны
create or replace function ratio_of_boats_from_country(_country text)
    returns integer
as
$$
declare
    from_country integer;
    all_boats    integer;
begin
    select count(id)
    into all_boats
    from boats;

    select count(boats.id)
    into from_country
    from boats
             inner join countries c on boats.country = c.id
    where c.name = _country;

    return round((from_country::double precision / all_boats::double precision) * 100);
end;
$$ language plpgsql;
--Найти долю яхт, поступивших из заданной страны
select ratio_of_boats_from_country('Нидерланды') || '%' as ratio;

--Найти все яхты с заданной вместительностью и заданным годом выпуска
select *
from boat_info
where roominess = 3 -- ввод вместимости
  and extract(year from (boat_info.date_of_issue)::timestamp) = 2010;
-- ввод года выпуска

--Найти все яхты заданного года выпуска, чья стоимость больше, чем средняя стоимость яхт, поступивших из заданной страны и заданными ограничениями по скорости
with avg_price as (select round(avg(price))
                   from boat_info
                   where country = 'США') -- ввод страны
select boat_info.*,
       (select * from avg_price) as avg_price
from boat_info
where max_speed between 150 and 250; -- ввод интервала скорости

--Определить количество ремонтируемых яхт, соотношение подлежащих ремонту и годных к эксплуатации (в процентах)
create or replace function boat_repairs_ratio()
    returns table
            (
                quantity_of_repairing int,
                ratio_of_repairable   int
            )
as
$$
declare
    repairing  int;
    repairable int;
    all_boats  int;
begin
    select count(id)
    into repairing
    from boats
    where status = 'Ремонтируется';

    select count(id)
    into all_boats
    from boats
    where status = 'Готова к эксплуатации';

    select count(boats.id)
    into repairable
    from boats
    where status = 'Подлежит ремонту';
    return query (select repairing, round((repairable::double precision / all_boats::double precision) * 100)::int);
end;
$$ language plpgsql;

--Определить количество ремонтируемых яхт, соотношение подлежащих ремонту и годных к эксплуатации (в процентах)
select quantity_of_repairing, ratio_of_repairable || '%' as ratio_of_repairable
from boat_repairs_ratio();

--Найти виды запасных частей, поступающих для ремонта судов
drop view if exists repairment_parts_info;
create or replace view repairment_parts_info as
select replacement_parts.name,
       replacement_parts.price,
       replacement_parts.vessel_type,
       rpp.name as provider,
       c.name   as provider_country
from replacement_parts
         inner join replacement_part_providers rpp on rpp.id = replacement_parts.provider
         inner join countries c on c.id = rpp.country;

--Найти виды запасных частей, поступающих для ремонта судов, отсортировать по наименованию
select *
from repairment_parts_info
order by name;

--Найти виды запасных частей, поступающих для ремонта судов, отсортировать по стоимости
select *
from repairment_parts_info
order by price;

--Найти виды запасных частей, поступающих для ремонта судов, отсортировать по поставщику
select *
from repairment_parts_info
order by provider;

--Найти поставщиков заданного вида запасных частей, для заданного вида яхт
select providers.name,
       rp.name,
       rp.vessel_type
from replacement_part_providers as providers
         inner join replacement_parts rp on providers.id = rp.provider
where rp.name = 'Стеклоочистители' -- ввод наименования запчасти
  and vessel_type = 'Скоростной катер';
-- ввод вида яхты

--Определить долю регулярных поставок заданного судна по заданной дате отгрузки от общего числа судов
create or replace function ratio_of_regular_supplies(_vessel_type vessel_type, date_of_supply date)
    returns integer
as
$$
declare
    regular_supply int;
    all_boats      int;
begin
    select count(id)
    into regular_supply
    from boats
    where date_of_sale = date_of_supply
      and vessel_type = _vessel_type;

    select count(id) into all_boats from boats where vessel_type = _vessel_type;

    return round((regular_supply::double precision / all_boats::double precision) * 100);
end;
$$ language plpgsql;

--Определить долю регулярных поставок заданного судна по заданной дате отгрузки от общего числа судов
select ratio_of_regular_supplies(
               'Скоростной катер', -- ввод вида яхты
               '2018-03-10' -- ввод даты отгрузки
           ) || '%' as ratio_of_regular_supplies;

--Найти самую популярную яхту (продано наибольшее количество)
select boats.vessel_type,    -- яхта
       count(id) as quantity -- количество проданных экземпляров
from boats
where date_of_sale is not null
group by vessel_type
order by quantity desc
limit 1;