create or replace view plumbings_info as
select p.id,
       pt.name,
       ps.name as provider,
       m.name  as manufacturer,
       c.name  as provider_country,
       c2.name as manufacturer_country,
       p.price,
       p.date_of_manufacture,
       p.date_of_supply,
       s.date  as date_of_sale
from plumbings as p
         inner join plumbing_types pt on pt.id = p.type
         inner join providers ps on ps.id = p.provider_id
         inner join manufacturers m on m.id = p.manufacturer_id
         inner join countries c2 on c2.id = m.country_id
         inner join countries c on c.id = ps.country_id
         left join sales s on p.id = s.plumbing_id;

--Для каждого вида сантехники указать сведения о ней (наименование, год выпуска, место изготовления, цена, поставщик и т.п.).
select *
from plumbings_info;

--Для каждого вида сантехники выдать список, отсортированный: по году выпуска, по поставщику, по стоимости.
select * --по году выпуска
from plumbings_info
order by date_of_manufacture;

select * --по поставщику
from plumbings_info
order by provider;

select * --по стоимости
from plumbings_info
order by price;

-- Найти самую дорогую сантехнику, самую дешевую, среднюю стоимость.
select max(price) as max_price,
       min(price) as min_price,
       avg(price) as avg_price
from plumbings_info;

--Найти сантехнику с ценой свыше 1000 р. (и любая другая сумма, т.е. предусмотреть ввод цены с клавиатуры).
select *
from plumbings_info
where price > 1000;

--Найти количество сантехники, выпущенной за определенный период (месяц, 3 месяца, 6 месяцев),среднюю стоимость, за этот же период - самую дорогую сантехнику, самую дешевую.
select count(*),
       avg(price) as avg_price,
       max(price) as max_price,
       min(price) as min_price
from plumbings_info
where date_of_manufacture between current_date - '4 years'::interval and current_date;

--Найти долю сантехники, поступившей из заданной страны (выбор страны) от общего числа сантехники.
with total as (select count(*) as total
               from plumbings_info),
     country as (select count(*) as country
                 from plumbings_info
                 where provider_country = 'Russia')
select round((country.country::double precision * 100.0 / total.total::double precision)) || '%' as ratio
from total,
     country;

--Найти всю сантехнику с заданной датой выпуска (ввод даты).
select *
from plumbings_info
where date_of_manufacture = '2018-10-14';

--Найти всю сантехнику заданного поставщика (ввод), чья стоимость находится в заданных пределах (ввод интервала).
select *
from plumbings_info
where provider = 'СанТехСити'
  and price between 100 and 1000;

--Найти долю сантехники, поступившей от заданного поставщика (ввод поставщика) от общего . числа поставщиков.
with total as (select count(*) as total
               from plumbings_info),
     provider as (select count(*) as provider
                  from plumbings_info
                  where provider = 'СанТехСити')
select round((provider.provider::double precision * 100.0 / total.total::double precision)) || '%' as ratio
from total,
     provider;

--Найти всю сантехнику заданного года выпуска, чья стоимость больше заданной (ввод стоимости).
select *
from plumbings_info
where extract(year from date_of_manufacture) = 2018
  and price > 1000;

--Найти всю сантехнику заданного производителя (выбор).
select *
from plumbings_info
where manufacturer = 'Kohler';

--Найти долю сантехники, проданной за определенный период (ввод периода) от общего времени продажи.
with total as (select count(*) as total
               from plumbings_info),
     period as (select count(*) as period
                from plumbings_info
                where date_of_sale between '2018-01-01' and '2019-01-01')
select round((period.period::double precision * 100.0 / total.total::double precision)) || '%' as ratio
from total,
     period;

--Найти всю сантехнику, поступившую от заданного поставщика (выбор поставщика), чья стоимость больше, чем средняя стоимость сантехники, поступившей из заданной страны (ввод страны).
with avg_price as (select round(avg(price)) as avg_price
                   from plumbings_info
                   where provider_country = 'Russia')
select plumbings_info.*,
       (select * from avg_price) as avg_price
from plumbings_info
where provider = 'СанТехСити'
  and price > (select * from avg_price);

--Найти долю дорогой сантехники (чья стоимость больше заданной, ввод стоимости), проданной заданному клиенту (ввод клиента), и в целом.
with total as (select count(*) as total -- заданному клиенту
               from plumbings_info
                        inner join sales on sales.plumbing_id = plumbings_info.id),
     client as (select count(*) as client
                from plumbings_info
                         inner join sales on sales.plumbing_id = plumbings_info.id
                         inner join clients c on c.id = sales.client_id
                where c.name = 'ИП Владислав Ширяев'
                  and price > 1000)
select round((client.client::double precision * 100.0 / total.total::double precision)) || '%' as ratio
from total,
     client;

with total as (select count(*) as total -- в целом
               from plumbings_info),
     client as (select count(*) as client
                from plumbings_info
                where price > 5500)
select round((client.client::double precision * 100.0 / total.total::double precision)) || '%' as ratio
from total,
     client;

--Найти среднюю стоимость сантехники, проданной за определенный промежуток времени (ввод интервала).
select round(avg(price)) as avg_price
from plumbings_info
where date_of_sale between '2018-01-01' and '2019-01-01';

--Найти всю сантехнику, чья стоимость выше, чем средняя стоимость сантехники заданного производителя (выбор)
with avg_price as (select round(avg(price)) as avg_price
                   from plumbings_info
                   where manufacturer = 'Kohler')
select plumbings_info.*,
       (select * from avg_price) as avg_price
from plumbings_info
where price > (select * from avg_price);

--Найти среднюю стоимость сантехники, поступившей от заданного поставщика (выбор поставщика).
select round(avg(price)) as avg_price
from plumbings_info
where provider = 'СанТехСити';

--Найти среднюю стоимость сантехники, поступившей от заданного поставщика (выбор поставщика), и в целом.
with provider as (select round(avg(price)) as avg_price
                  from plumbings_info
                  where provider = 'СанТехСити'),
     total as (select round(avg(price)) as avg_price
               from plumbings_info)
select (select * from provider) as provider_avg_price,
       (select * from total) as total_avg_price;

--Найти сантехнику с наибольшей стоимостью.
select *
from plumbings_info
where price = (select max(price)
               from plumbings_info);

--Найти сантехнику с наименьшей стоимостью.
select *
from plumbings_info
where price = (select min(price)
               from plumbings_info);

--Найти сантехнику с наибольшей стоимостью, поступившей от заданного поставщика (выбор поставщика).
select *
from plumbings_info
where price = (select max(price)
               from plumbings_info
               where provider = 'СанТехСити');
