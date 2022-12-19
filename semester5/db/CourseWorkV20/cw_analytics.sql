create or replace view ice_creams_info as
select s.serving,
       t.taste,
       ic.price,
       ic.weight,
       ic.is_defective,
       ic.date_of_issue,
       m.name  as manufacturer,
       mc.name as manufacturer_city,
       ic.date_of_supply,
       p.name  as provider,
       ic.date_of_sale
from ice_creams as ic
         inner join tastes as t on ic.taste = t.id
         inner join servings s on s.id = ic.serving
         inner join manufacturers m on m.id = ic.manufacturer_id
         inner join cities mc on mc.id = m.city_id
         inner join providers p on ic.provider_id = p.id;

--Для каждого вида мороженого указать сведения о нем (наименование, дата выпуска, место изготовления, цена, вес и т.п.).
select *
from ice_creams_info;

--Для каждого вида товара указать сведения о поставщике (наименование, процент бракованной продукции).
select ic.taste,
       ic.serving,
       p.name,
       ((select count(*)
         from ice_creams
         where provider_id = p.id
           and is_defective = true) * 100 / (select count(*)
                                             from ice_creams
                                             where provider_id = p.id)) || '%' as defective_percent
from ice_creams_info as ic
         inner join providers as p on ic.provider = p.name;

--Для каждого вида мороженого выдать список, отсортированный: по дате выпуска, в алфавитном порядке, по весу, по стоимости.
select * -- по дате выпуска
from ice_creams_info
order by date_of_issue;

select * -- в алфавитном порядке
from ice_creams_info
order by taste, serving;

select * -- по весу
from ice_creams_info
order by weight;

select * -- по стоимости
from ice_creams_info
order by price;

--Найти самое дорогое мороженое, самое дешевое, среднюю стоимость.
select max(price) as max_price,
       min(price) as min_price,
       avg(price) as avg_price
from ice_creams_info;

--Найти мороженое с ценой свыше 5 р. (и любая другая сумма, т.е. предусмотреть ввод цены с клавиатуры).
select *
from ice_creams_info
where price > 50;

--Найти количество мороженого, проданного за определенный период (месяц, 3 месяца. 6 месяцев), среднюю стоимость, за этот же период - самое дорогое мороженое, самое дешевое, вес
select count(*)           as count,
       round(avg(price))  as avg_price,
       max(price)         as max_price,
       min(price)         as min_price,
       sum(weight)        as sum_weight,
       round(avg(weight)) as avg_weight
from ice_creams_info
where date_of_sale between '2021-01-01' and '2021-03-31';

--Найти долю мороженого, поступившего от заданного поставщика (выбор поставшика) от общего числа поставщиков.
with total as (select count(*) as total
               from ice_creams_info),
     selected as (select count(*) as selected
                  from ice_creams_info
                  where provider = 'Гулливер')
select round((selected.selected::double precision * 100.0) / total.total::double precision) || '%' as percent
from total,
     selected;

--Найти все мороженое с заданной датой выпуска (ввод даты).
select *
from ice_creams_info
where date_of_issue = '2020-01-02';

--Найти все мороженое с заданным весом (ввод), чья стоимость находится в заданных пределах. (ввод интервала).
select *
from ice_creams_info
where weight = 100
  and price between 50 and 100;

--Найти все мороженое, поступившие от заданного поставщика (ввод поставщика), чья стоимость больше заданной (ввод стоимости).
select *
from ice_creams_info
where provider = 'Гулливер'
  and price > 50;

--Найти все мороженое заданного производителя (выбор производителя).
select *
from ice_creams_info
where manufacturer = 'Вилон';

--Найти долю дешевого мороженого (меньше заданного, ввод ограничения) от общего числа мороженого.
with total as (select count(*) as total
               from ice_creams_info),
     selected as (select count(*) as selected
                  from ice_creams_info
                  where price < 50)
select round((selected.selected::double precision * 100.0) / total.total::double precision) || '%' as percent
from total,
     selected;

--Определить долю регулярных поставок заданного мороженого по заданной дате отгрузки (ввод) от общего числа мороженого.
create or replace function ratio_of_regular_supplies(_taste text, _serving text, _date date) returns double precision
as
$$
declare
    target       double precision;
    all_products double precision;
begin
    select count(*)::double precision
    into target
    from ice_creams_info
    where ice_creams_info.taste = _taste
      and extract(month from (ice_creams_info.date_of_supply)::timestamp) = extract(month from _date::timestamp);
    select count(*)
    into all_products
    from ice_creams_info
    where ice_creams_info.taste = _taste;
    return round((target / all_products) * 100::double precision);
end;
$$
    language plpgsql;

select ratio_of_regular_supplies('Клубничный', 'Рожок', '2021-01-01') || '%' as ratio;

--Найти количество бракованного мороженого, поступившего от заданного поставщика (выбор поставщика).
select count(*) as count
from ice_creams_info
where manufacturer = 'Вилон'
  and is_defective = true;

--Найти все мороженое, поступившее от заданного поставщика (ввод поставщика), чья стоимость больше, чем средняя стоимость мороженого, поступившего из заданного города (ввод города).
select *
from ice_creams_info
where provider = 'Гулливер'
  and price > (select avg(price)
               from ice_creams_info
               where manufacturer_city = 'Курск');