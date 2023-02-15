drop schema if exists public cascade;
create schema public;

create table ad_types
(
    id   serial primary key,
    type text unique
);

insert into ad_types (type)
values ('Сниму'),
       ('Сдам'),
       ('Продам'),
       ('Куплю');

create table cities
(
    id   serial primary key,
    name text unique
);

insert into cities (name)
values ('Санкт-Петербург'),
       ('Москва'),
       ('Краснодар'),
       ('Владивосток'),
       ('Новосибирск');

create table ads
(
    id         serial primary key,
    type       int references ad_types (id),
    city       int references cities (id),
    address    text,
    roominess  int,
    price      int,
    created_at timestamp default now()
);

insert into ads (type, city, address, roominess, price)
values (1, 1, 'ул. Красная, 1', 2, 20000),
       (2, 2, 'ул. Синяя, 2', 1, 20000),
       (3, 3, 'ул. Зеленая, 3', 3, 8000000),
       (4, 4, 'ул. Карла Маркса, 2', 4, 40000000),
       (1, 5, 'ул. Зорге, 21', 2, 25000),
       (4, 4, 'ул. Желтая, 4', 4, 40000);

select
    ads.id,
    type.type,
    city.name as city,
    ads.address,
    ads.roominess,
    ads.price,
    ads.created_at
from ads
         inner join ad_types as type on ads.type = type.id
         inner join cities as city on ads.city = city.id;

