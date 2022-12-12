create table product
(
    id_product  serial      not null primary key,
    name        varchar(20) not null,
    id_type     int references types (id_type) on delete restrict,
    id_producer int references producer (id) on delete restrict,
    count       numeric,
    cost        numeric
);
create table sales
(
    id_product serial not null primary key,
    price      numeric,
    count      numeric,
    date       date
);
create table city
(
    id           serial             not null primary key,
    city         varchar(20) unique not null,
    country_name varchar(20)
);

create table producer
(
    id            serial      not null primary key,
    producer_name varchar(50) not null,
    city          int references city (id) on delete restrict
);

select producer_name, sum(profit) as sum_profit
from (select producer_name, (sales.price - product.cost) * sales.count as profit
      from producer
               join product on producer.id = product.id_producer
               join sales on product.id_product = sales.id_product
      group by producer_name, product.name, sales.price, product.cost, sales.count) as nastya
group by producer_name;