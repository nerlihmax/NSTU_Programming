create or replace view product_info as
select ppt.name,
       m.name as manufacturer,
       p.name as provider,
       c.name as supply_city,
       product.date_of_manufacture,
       product.date_of_sale,
       product.price,
       product.is_defective
from porcelain_products as product
         inner join porcelain_product_types ppt on ppt.id = product.porcelain_product_type_id
         inner join manufacturers m on m.id = product.manufacturer_id
         inner join providers p on p.id = product.provider_id
         inner join cities c on p.city_id = c.id;

--Для каждого изделия из фарфора указать сведения о нем (наименование, дату выпуска, поставщик, цена и т.п.).
select *
from product_info;

--Получить список изделий из фарфора, отсортированный: по дате выпуска, по наименованию, по стоимости.
select * --по дате выпуска
from product_info
order by date_of_manufacture;

select * --по наименованию
from product_info
order by name;

select * --по стоимости
from product_info
order by price;

-- Найти самое дорогое изделие из фарфора, самое дешевое излелие из фарфора, среднюю стоимость.
select max(price) as max_price,
       min(price) as min_price,
       avg(price) as avg_price
from product_info;

--Найти изделия из фарфора с ценой свыше 6000 р. (и любая другая сумма, т.е. предусмотреть ввод цены с клавиатуры).
select *
from product_info
where price > 6000;

--Найти долю изделий из фарфора заданного производителя (выбор) от общего числа издели
with total as (select count(*) as count
               from product_info),
     manufacturer as (select count(*) as count
                      from product_info
                      where manufacturer = 'Кузнецовский фарфор')
select round((manufacturer.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     manufacturer;

--Найти все изделия из фарфора с заданной датой выпуска (ввод даты).
select *
from product_info
where date_of_manufacture = '2018-06-03';

--Найти доло изделий из фарфора, чья стоимость находится в заданных пределах (ввод интервала) от общего количества изделий.
with total as (select count(*) as count
               from product_info),
     price as (select count(*) as count
               from product_info
               where price between 1000 and 2000)
select round((price.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     price;

--Найти долю изделий из фарфора, поступивших от заданного поставщика (выбор поставщика) от общего числа поставшиков.
with total as (select count(*) as count
               from product_info),
     provider as (select count(*) as count
                  from product_info
                  where provider = 'Современник')
select round((provider.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     provider;

--Найти все изделия из фарфора, поступившие от заданного поставщика (выбор поставщика), чья стоимость больше заданной (ввод стоимости).
select *
from product_info
where provider = 'Современник'
  and price > 5000;

--Найти все изделия из фарфора заданного производителя (выбор).
select *
from product_info
where manufacturer = 'Кузнецовский фарфор';

--Найти долю изделий из фарфора, проданных за определенный период (ввод периода) от общего времени продажи
with total as (select count(*) as count
               from product_info
               where date_of_sale is not null),
     sales as (select count(*) as count
               from product_info
               where date_of_sale between '2019-01-01' and '2019-12-31')
select round((sales.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     sales;

--Найти все изделия из фарфора, поступившие от заданного поставщика, чья стоимость больше, чем средняя стоимость изделий из фарфора, поступивших из заданного города (выбор город
with avg_price as (select avg(price) as avg_price
                   from product_info
                   where supply_city = 'Екатеринбург')
select product_info.*, (select * from avg_price) as avgp
from product_info
where product_info.provider = 'Современник'
  and product_info.price > (select * from avg_price);

--Найти долю дешевых изделий из фарфора (чья стоимость меньше заданной, ввод стоимости), поступивших от заданного поставщика и в целом.
with total as (select count(*) as count -- в целом
               from product_info),
     cheap as (select count(*) as count
               from product_info
               where price < 1000)
select round((cheap.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     cheap;

with total as (select count(*) as count -- от поставщика
               from product_info),
     cheap as (select count(*) as count
               from product_info
               where price < 1000
                 and provider = 'Современник')
select round((cheap.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     cheap;

--Найти среднюю стоимость изделий из фарфора, проданных за определенный промежуток времени (ввод интервала).
select round(avg(price)) as avg_price
from product_info
where date_of_sale between '2019-01-01' and '2019-12-31';

--Найти все изделия из фарфора, чья стоимость выше, чем средняя стоимость изделий из фарфора заданного производителя (выбор).
with avg_price as (select avg(price) as avg_price
                   from product_info
                   where manufacturer = 'Кузнецовский фарфор')
select product_info.*, round((select avg_price from avg_price)) as avg_price
from product_info
where product_info.price > (select * from avg_price);

--Найти процент бракованных изделий из фарфора для заданного поставщика (выбор поставщика).
with total as (select count(*) as count
               from product_info
               where provider = 'Дизайнер'),
     broken as (select count(*) as count
                from product_info
                where provider = 'Дизайнер'
                  and is_defective = true)
select round((broken.count::double precision / total.count::double precision) * 100) || '%' as ratio
from total,
     broken;
